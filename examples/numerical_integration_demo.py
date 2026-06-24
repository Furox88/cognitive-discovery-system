"""Deterministic quadrature demo — integrate e^x on [0,1]."""

import math

from cds.numerical_integration import (
    adaptive_simpson,
    gaussian_quadrature,
    gaussian_quadrature_2d,
    romberg,
    simpson,
    simpson_2d,
    simpson_38,
    trapezoid,
)


def main() -> None:
    f = math.exp
    exact = math.e - 1  # ≈ 1.718281828...

    print("=== ∫_0^1 e^x dx = e - 1 ≈ 1.718281828 ===\n")
    print(f"{'method':<28}{'value':>14}{'abs err':>14}")
    print("-" * 56)
    # trapezoid / simpson / gaussian_quadrature return plain floats.
    for name, v in [
        ("Trapezoid (n=1000)", trapezoid(f, 0, 1, 1000)),
        ("Simpson 1/3 (n=100)", simpson(f, 0, 1, 100)),
        ("Simpson 3/8 (n=99)", simpson_38(f, 0, 1, 99)),
        ("Gauss-Legendre (n=8)", gaussian_quadrature(f, 0, 1, 8)),
    ]:
        print(f"{name:<28}{v:>14.10f}{abs(v - exact):>14.2e}")

    # romberg / adaptive_simpson return a QuadratureResult (has .value).
    r = romberg(f, 0, 1)
    print(f"{'Romberg (auto tol)':<28}{r.value:>14.10f}{abs(r.value - exact):>14.2e}")

    a = adaptive_simpson(f, 0, 1)
    print(f"{'Adaptive Simpson':<28}{a.value:>14.10f}{abs(a.value - exact):>14.2e}")

    # --- 2-D tensor-product quadrature -------------------------------------
    print("\n=== ∬_[0,1]^2 e^{x+y} dx dy = (e-1)^2 ≈ 2.952492442 ===\n")
    g = lambda x, y: math.exp(x + y)  # noqa: E731
    exact_2d = (math.e - 1) ** 2
    print(f"{'method':<32}{'value':>16}{'abs err':>14}")
    print("-" * 62)
    s2 = simpson_2d(g, 0, 1, 0, 1, 50, 50)
    print(f"{'Simpson 2-D (50x50)':<32}{s2:>16.10f}{abs(s2 - exact_2d):>14.2e}")
    g2 = gaussian_quadrature_2d(g, 0, 1, 0, 1, 5)
    print(f"{'Gauss-Legendre 2-D (n=5)':<32}{g2:>16.10f}{abs(g2 - exact_2d):>14.2e}")

    # Polynomial exactness: 3 nodes/axis integrate x^5 y^5 exactly (1/36).
    poly = gaussian_quadrature_2d(lambda x, y: x**5 * y**5, 0, 1, 0, 1, 3)
    print(
        f"\nGauss-Legendre 2-D (n=3) on x^5 y^5 = {poly:.12f}"
        f"  (exact 1/36 = {1/36:.12f}, err {abs(poly - 1/36):.2e})"
    )


if __name__ == "__main__":
    main()
