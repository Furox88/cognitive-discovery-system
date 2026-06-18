"""Monte Carlo methods demo — pi estimation, integration, random walks, Buffon."""

import math

from cds.montecarlo import (
    buffon_needle,
    estimate_pi,
    mc_integrate,
    random_walk_1d,
    random_walk_2d,
)


def main() -> None:
    print("=== π Estimation ===")
    result = estimate_pi(n_samples=100_000, seed=42)
    print(f"estimate = {result.estimate:.6f}")
    print(f"true π  = {math.pi:.6f}")
    print(f"error   = {abs(result.estimate - math.pi):.5f}")

    print("\n=== Monte Carlo Integration: ∫_0^1 x^2 dx = 1/3 ===")
    integral = mc_integrate(lambda x: x**2, a=0.0, b=1.0, n_samples=50_000, seed=7)
    print(f"estimate = {integral.estimate:.6f}  (true = 1/3 ≈ 0.333333)")

    print("\n=== 1D Random Walk ===")
    w1 = random_walk_1d(steps=100, seed=1)
    print(f"positions (first 5): {w1[:5]}")
    print(f"final position after 100 steps (seed=1): {w1[-1]:.4f}")

    print("\n=== 2D Random Walk ===")
    w2 = random_walk_2d(steps=100, seed=2)
    print(f"positions (first 3): {w2[:3]}")
    print(f"final (x, y) after 100 steps (seed=2): ({w2[-1][0]:.4f}, {w2[-1][1]:.4f})")

    print("\n=== Buffon's Needle ===")
    bn = buffon_needle(needle_length=0.5, line_spacing=1.0, n_throws=10_000, seed=3)
    print(f"π via Buffon = {bn.estimate:.4f}")


if __name__ == "__main__":
    main()
