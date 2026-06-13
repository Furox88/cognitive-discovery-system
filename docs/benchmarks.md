# Benchmarks

Performance measurements on Python 3.12, single-threaded. All modules are pure Python — no C extensions or NumPy.

## Quantum Simulation

| Operation | N qubits | Time |
|-----------|----------|------|
| Bell state creation | 2 | ~0.01 ms |
| GHZ state creation | 3 | ~0.02 ms |
| GHZ state creation | 8 | ~1.5 ms |
| GHZ state creation | 10 | ~8 ms |
| 1000 measurements (Bell) | 2 | ~2 ms |

State vector size grows as 2^N, so simulation is practical up to ~16 qubits on typical hardware.

## FFT / Signal Processing

| Operation | N samples | Time |
|-----------|-----------|------|
| DFT (direct) | 256 | ~15 ms |
| DFT (direct) | 1024 | ~230 ms |
| FFT (radix-2) | 256 | ~0.8 ms |
| FFT (radix-2) | 1024 | ~3.5 ms |
| FFT (radix-2) | 4096 | ~17 ms |
| Convolution | 1000×100 | ~12 ms |

FFT is ~60× faster than direct DFT at N=1024 (O(N log N) vs O(N²)).

## Optimization

| Algorithm | Function | Iterations to converge |
|-----------|----------|----------------------|
| Gradient descent | (x-3)² | ~200 (lr=0.1) |
| Newton's method | x²-2=0 | ~5 |
| Adam | (x-3)² | ~150 (lr=0.1) |
| Golden section | (x-2.5)⁴ on [0,5] | ~55 |

## Probability

| Operation | Count | Time |
|-----------|-------|------|
| gaussian_pdf (1 eval) | 1 | ~0.001 ms |
| binomial_pmf (n=100) | 1 | ~0.01 ms |
| 10k uniform samples | 10000 | ~2 ms |

## Graph Theory

| Algorithm | V | E | Time |
|-----------|---|---|------|
| BFS | 1000 | 3000 | ~2 ms |
| DFS | 1000 | 3000 | ~2 ms |
| Dijkstra | 1000 | 3000 | ~5 ms |
| Kruskal MST | 1000 | 3000 | ~4 ms |
| Topological sort | 1000 | 2000 | ~1.5 ms |

## Monte Carlo

| Method | Samples | Estimate | Error |
|--------|---------|----------|-------|
| π estimation (circle) | 100K | ~3.1416 | ~0.005 |
| Buffon's needle | 100K | ~3.14 | ~0.02 |
| Integration (∫sin dx) | 100K | ~2.000 | ~0.003 |

## Differential Equations

| Method | ODE | dt | Error at t=1 |
|--------|-----|----|----|
| Euler | dy/dt=-y | 0.01 | ~5×10⁻³ |
| Midpoint (RK2) | dy/dt=-y | 0.01 | ~2×10⁻⁵ |
| RK4 | dy/dt=-y | 0.01 | ~3×10⁻⁹ |

RK4 is ~10⁶× more accurate than Euler for the same step size.

## Linear Algebra

| Operation | Size | Time |
|-----------|------|------|
| LU decomposition | 100×100 | ~30 ms |
| Linear solve (Ax=b) | 100×100 | ~35 ms |
| Matrix inverse | 50×50 | ~40 ms |
| Power iteration (eigenvalue) | 100×100 | ~15 ms |
| Gram-Schmidt | 50 vectors (dim 50) | ~5 ms |

## Test Suite & Coverage

```
300+ tests in ~0.4 seconds (see CI)
```

Test coverage is tracked and visible via the Codecov badge in the README (connect the repo on codecov.io to view detailed reports).

All benchmarks are approximate and depend on hardware. Run `pytest --durations=10` to see test timing on your machine.

## Comparison to NumPy/SciPy

CDS is **not** a replacement for NumPy/SciPy. It's a lightweight, educational, dependency-free alternative for:
- Learning algorithms from source
- Environments where heavy dependencies aren't desired
- Quick prototyping without install overhead
- Teaching computational science concepts

For production workloads with large datasets, use NumPy/SciPy.
