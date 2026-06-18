"""Calculus and linear algebra demo."""

import math

from cds.math_utils import (
    derivative,
    determinant,
    dot,
    gradient,
    integral,
    lu_decomposition,
    mat_mul,
    power_iteration,
    qr_decomposition,
    solve_linear,
    transpose,
)


def main() -> None:
    print("=== Calculus ===")
    d = derivative(lambda x: x**2, x=3.0)
    i = integral(lambda x: math.sin(x), a=0.0, b=math.pi)
    g = gradient(lambda x, y: x**2 + y**2, point=[1.0, 2.0])
    print(f"d/dx(x^2) @ x=3  = {d:.4f}  (expect 6)")
    print(f"∫_0^π sin(x) dx   = {i:.4f}  (expect 2)")
    print(f"grad(x^2+y^2)@[1,2] = {g}")

    print("\n=== Linear Algebra ===")
    A = [[2.0, 1.0], [1.0, 3.0]]
    eye = [[1.0, 0.0], [0.0, 1.0]]
    print(f"mat_mul(A, I) = {mat_mul(A, eye)}")
    print(f"transpose(A)  = {transpose(A)}")
    print(f"determinant(A)= {determinant(A):.4f}  (expect 5)")
    L, U, P = lu_decomposition(A)
    print(f"LU L = {L}")
    print(f"LU U = {U}")
    x = solve_linear(A, [3.0, 4.0])
    print(f"solve A·x=[3,4]: x = {x}")

    print("\n=== Decompositions & Spectral ===")
    Q, R = qr_decomposition(A)
    print(f"QR Q = {Q}")
    print(f"QR R = {R}")
    eigval, eigvec = power_iteration(A)
    print(f"power_iteration dominant eigenvalue = {eigval:.6f}")

    print("\n=== Vector ops ===")
    print(f"dot([1,2],[3,4]) = {dot([1.0, 2.0], [3.0, 4.0])}  (expect 11)")


if __name__ == "__main__":
    main()
