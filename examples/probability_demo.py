"""Probability distributions and sampling demo."""

import math

from cds.probability import (
    binomial_pmf,
    exponential_pdf,
    gaussian_pdf,
    poisson_pmf,
    uniform_pdf,
    uniform_sample,
)


def main() -> None:
    print("=== Continuous PDFs ===")
    peak = gaussian_pdf(0.0, mu=0.0, sigma=1.0)
    print(f"gaussian_pdf(0; mu=0, sigma=1) = {peak:.6f}")
    print(f"  (1/sqrt(2π) ≈ {1.0 / math.sqrt(2 * math.pi):.6f})")
    print(f"uniform_pdf(0.5; a=0, b=1) = {uniform_pdf(0.5, a=0.0, b=1.0):.6f}  (expect 1)")
    print(f"exponential_pdf(1; lam=2) = {exponential_pdf(1.0, lam=2.0):.6f}")

    print("\n=== Discrete PMFs ===")
    print("Binomial PMF (n=10, p=0.5):")
    for k in range(0, 11):
        print(f"  P(X={k}) = {binomial_pmf(k, n=10, p=0.5):.4f}")
    print("Poisson PMF (lam=3):")
    for k in range(0, 6):
        print(f"  P(X={k}) = {poisson_pmf(k, lam=3.0):.4f}")

    print("\n=== Sampling ===")
    samples = uniform_sample(a=0.0, b=1.0, n=5, seed=42)
    print(f"5 samples from Uniform(0,1), seed=42: {samples}")
    mean = sum(samples) / len(samples)
    print(f"sample mean = {mean:.4f}")


if __name__ == "__main__":
    main()
