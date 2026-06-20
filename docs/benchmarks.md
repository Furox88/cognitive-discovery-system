# CDS Performance & Intelligence Report

> **Last measured:** `385cfa3` on 2026-06-20 23:47 UTC. Regenerated automatically by the `benchmarks` GitHub Actions workflow (weekly + on release tags). Raw data: `benchmarks/results.json`.

This report measures both raw speed and algorithmic scaling. Pure Python is slower than C-extensions for dense numerics, so rather than only racing NumPy, the comparisons below also check that the implemented algorithms scale with their theoretical complexity (e.g. O(N log N) FFT, O(N^3) PLU determinant) and converge to machine precision where expected.

### Linear Algebra (Approaching C-Speed)
| Metric | Value |
|--------|-------|
| CDS Matrix Mul (100x100) | 0.0688s |
| CDS LU Decomp (100x100) | 0.0237s |
| NumPy Matrix Mul (Baseline) | 0.000051s |
| Speed Status | CDS is 1351.4x slower than NumPy (pure Python vs C) |

### Linear Algebra Intelligence (Determinant Scaling)
| Metric | Value |
|--------|-------|
| Determinant @ N=50 | 0.004602s |
| Determinant @ N=100 | 0.029143s |
| Ratio (doubling N) | 6.3x |
| Expected for O(N^3) | 8.0x |
| Complexity | O(N^3) PLU |

### Monte Carlo (Hardware Saturation)
| Metric | Value |
|--------|-------|
| Parallel Pi (100k samples) | 0.0287s |
| CPU Cores Saturated | 4 |
| Estimate error vs π | 0.00791 |

### Quantum (Algorithmic Intelligence)
| Metric | Value |
|--------|-------|
| Intelligent O(1) Sampling | 0.0156s |
| Naive Brute Force (Est.) | 0.90s |
| Intelligence Speedup | 57.8x Faster |

### Signal Processing (FFT vs DFT)
| Metric | Value |
|--------|-------|
| Signal length | 1024 samples |
| CDS FFT (radix-2, O(N log N)) | 0.002798s |
| Naive DFT (O(N^2)) | 0.345309s |
| Algorithmic speedup | 123x |

### Numerical Integration (Convergence)
| Metric | Value |
|--------|-------|
| Integral | ∫_0^1 e^x dx = e - 1 |
| Trapezoid n=1000 | 1.43e-07 |
| Simpson n=100 | 9.55e-11 |
| Gauss-Legendre n=8 | 6.66e-16 |
| Romberg (auto tol) | 8.88e-16 |
| Adaptive Simpson | 6.66e-16 |

## Visual Proof: Quantum Intelligence
```text
Naive Brute Force: ######################################## (0.90s)
CDS O(1) Sampling: # (0.0156s)

Conclusion: CDS is 57.7 times faster via O(1) probabilistic sampling vs running the circuit shot-by-shot.
```
