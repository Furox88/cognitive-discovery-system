"""Numerical linear algebra demo (QR, Cholesky, eigenvalues, solving)."""
from cds.math_utils import (
    cholesky,
    mat_mul,
    matrix_inverse,
    power_iteration,
    qr_decomposition,
    solve_linear,
    transpose,
)

# --- QR decomposition (Householder, 1958) ---
print("=== QR decomposition (Householder) ===")
A = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
Q, R = qr_decomposition(A)
print("Q (orthogonal):")
for row in Q:
    print("  " + "  ".join(f"{x:7.4f}" for x in row))
print("R (upper triangular):")
for row in R:
    print("  " + "  ".join(f"{x:9.4f}" for x in row))

# --- Cholesky decomposition (Benoit/Cholesky, 1924) ---
print("\n=== Cholesky decomposition ===")
M = [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
L = cholesky(M)
print("L (lower triangular), A = L Lᵀ:")
for row in L:
    print("  " + "  ".join(f"{x:6.2f}" for x in row))

# --- Dominant eigenvalue (power iteration, Von Mises 1929) ---
print("\n=== Power iteration ===")
S = [[2.0, 1.0], [1.0, 3.0]]
eigval, eigvec = power_iteration(S)
print(f"Dominant eigenvalue: {eigval:.6f}")
print(f"Eigenvector:         [{eigvec[0]:.4f}, {eigvec[1]:.4f}]")

# --- Solving a linear system ---
print("\n=== Solve Ax = b ===")
A2 = [[3.0, 2.0], [1.0, 2.0]]
b = [12.0, 8.0]
x = solve_linear(A2, b)
print(f"Solution x = [{x[0]:.4f}, {x[1]:.4f}]")

# --- Matrix inverse ---
print("\n=== Matrix inverse (A · A⁻¹ = I) ===")
inv = matrix_inverse(A2)
check = mat_mul(A2, inv)
print("A · A⁻¹ =")
for row in check:
    print("  " + "  ".join(f"{x:7.4f}" for x in row))

_ = transpose  # exported helper available for compositions
