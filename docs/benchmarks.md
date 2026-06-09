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

## Test Suite

```
162+ tests in ~0.2 seconds
```

All benchmarks are approximate and depend on hardware. Run `pytest --durations=10` to see test timing on your machine.

## Comparison to NumPy/SciPy

CDS is **not** a replacement for NumPy/SciPy. It's a lightweight, educational, dependency-free alternative for:
- Learning algorithms from source
- Environments where heavy dependencies aren't desired
- Quick prototyping without install overhead
- Teaching computational science concepts

For production workloads with large datasets, use NumPy/SciPy.
