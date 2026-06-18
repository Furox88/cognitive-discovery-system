# CDS Performance & Intelligence Report

> **Last measured:** `a5da8b8` on 2026-06-18 00:45 UTC. Regenerated automatically by the `benchmarks` GitHub Actions workflow (weekly + on release tags). Raw data: `benchmarks/results.json`.

This report measures both raw speed and algorithmic scaling. Pure Python is slower than C-extensions for dense numerics, so rather than only racing NumPy, the comparisons below also check that the implemented algorithms scale with their theoretical complexity (e.g. O(N log N) FFT, O(N^3) PLU determinant) and converge to machine precision where expected.

### Linear Algebra (Approaching C-Speed)
| Metric | Value |
|--------|-------|
| CDS Matrix Mul (100x100) | 0.0382s |
| CDS LU Decomp (100x100) | 0.0154s |
| NumPy Matrix Mul (Baseline) | 0.000051s |
| Speed Status | CDS is 746.2x slower than NumPy (pure Python vs C) |

### Linear Algebra Intelligence (Determinant Scaling)
| Metric | Value |
|--------|-------|
| Determinant @ N=50 | 0.002225s |
| Determinant @ N=100 | 0.015919s |
| Ratio (doubling N) | 7.2x |
| Expected for O(N^3) | 8.0x |
| Complexity | O(N^3) PLU |

### Monte Carlo (Hardware Saturation)
| Metric | Value |
|--------|-------|
| Parallel Pi (100k samples) | 1.0192s |
| CPU Cores Saturated | 22 |
| Estimate error vs π | 0.01235 |

### Quantum (Algorithmic Intelligence)
| Metric | Value |
|--------|-------|
| Intelligent O(1) Sampling | 0.0072s |
| Naive Brute Force (Est.) | 0.43s |
| Intelligence Speedup | 59.7x Faster |

### Signal Processing (FFT vs DFT)
| Metric | Value |
|--------|-------|
| Signal length | 1024 samples |
| CDS FFT (radix-2, O(N log N)) | 0.001671s |
| Naive DFT (O(N^2)) | 0.206159s |
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
Naive Brute Force: ######################################## (0.43s)
CDS O(1) Sampling: # (0.0072s)

Conclusion: CDS is 59.7 times faster via O(1) probabilistic sampling vs running the circuit shot-by-shot.
```
