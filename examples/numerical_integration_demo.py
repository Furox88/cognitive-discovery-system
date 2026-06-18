"""Deterministic quadrature demo — integrate e^x on [0,1]."""

import math

from cds.numerical_integration import (
    adaptive_simpson,
    gaussian_quadrature,
    romberg,
    simpson,
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


if __name__ == "__main__":
    main()
