"""Linear algebra — pure Python, no dependencies.

References:
    - Golub, G.H. & Van Loan, C.F. Matrix Computations (4th ed.)
    - Trefethen, L.N. & Bau, D. Numerical Linear Algebra.
    - Von Mises, R. (1929). Praktische Verfahren der Gleichungsauflösung (power iteration).
"""
from __future__ import annotations

import math

Matrix = list[list[float]]
Vector = list[float]


def dot(a: Vector, b: Vector) -> float:
    if len(a) != len(b):
        raise ValueError("vectors must be same length")
    return sum(x * y for x, y in zip(a, b))


def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError(f"incompatible shapes: {rows_a}x{cols_a} and {rows_b}x{cols_b}")
    result: Matrix = [[0.0] * cols_b for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result


def transpose(m: Matrix) -> Matrix:
    if not m:
        return []
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def determinant(m: Matrix) -> float:
    n = len(m)
    if n == 1:
        return m[0][0]
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    det = 0.0
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in m[1:]]
        det += ((-1) ** j) * m[0][j] * determinant(minor)
    return det


def identity(n: int) -> Matrix:
    """Create n×n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def lu_decomposition(m: Matrix) -> tuple[Matrix, Matrix]:
    """LU decomposition without pivoting (Doolittle's method).

    A = L * U where L is lower triangular (ones on diagonal)
    and U is upper triangular.
    [Golub & Van Loan, §3.2]

    Raises:
        ValueError: if matrix is singular (zero pivot)
    """
    n = len(m)
    L = identity(n)
    U = [row[:] for row in m]

    for k in range(n):
        if abs(U[k][k]) < 1e-15:
            raise ValueError("zero pivot encountered — matrix may be singular")
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]

    return L, U


def solve_linear(A: Matrix, b: Vector) -> Vector:
    """Solve Ax = b using LU decomposition.

    Forward substitution for Ly = b, then back substitution for Ux = y.

    Raises:
        ValueError: if matrix is singular
    """
    n = len(A)
    L, U = lu_decomposition(A)

    # forward: Ly = b
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i] - sum(L[i][j] * y[j] for j in range(i))

    # backward: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < 1e-15:
            raise ValueError("singular matrix")
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return x


def matrix_inverse(m: Matrix) -> Matrix:
    """Compute matrix inverse via LU decomposition.

    Solves A * X = I column by column.

    Raises:
        ValueError: if matrix is singular
    """
    n = len(m)
    inv = identity(n)
    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0
        x = solve_linear(m, e)
        for row in range(n):
            inv[row][col] = x[row]
    return inv


def power_iteration(
    m: Matrix, max_iter: int = 1000, tol: float = 1e-10,
) -> tuple[float, Vector]:
    """Find dominant eigenvalue and eigenvector using power iteration.

    Von Mises iteration (1929). Converges to the largest eigenvalue
    in absolute value and its corresponding eigenvector.

    Args:
        m: square matrix
        max_iter: iteration limit
        tol: convergence tolerance

    Returns:
        (eigenvalue, eigenvector) tuple
    """
    n = len(m)
    v = [1.0] * n
    norm = math.sqrt(sum(x * x for x in v))
    v = [x / norm for x in v]

    eigenvalue = 0.0
    for _ in range(max_iter):
        # w = A * v
        w = [sum(m[i][j] * v[j] for j in range(n)) for i in range(n)]
        new_eigenvalue = sum(w[i] * v[i] for i in range(n))
        norm = math.sqrt(sum(x * x for x in w))
        if norm < 1e-15:
            break
        v_new = [x / norm for x in w]

        if abs(new_eigenvalue - eigenvalue) < tol:
            return new_eigenvalue, v_new
        eigenvalue = new_eigenvalue
        v = v_new

    return eigenvalue, v


def gram_schmidt(vectors: list[Vector]) -> list[Vector]:
    """Gram-Schmidt orthonormalization.

    Produces an orthonormal set from the input vectors.
    [Trefethen & Bau, Lecture 8]

    Args:
        vectors: list of linearly independent vectors

    Returns:
        orthonormal basis vectors
    """
    ortho: list[Vector] = []
    for v in vectors:
        u = v[:]
        for q in ortho:
            proj = sum(u[i] * q[i] for i in range(len(u)))
            u = [u[i] - proj * q[i] for i in range(len(u))]
        norm = math.sqrt(sum(x * x for x in u))
        if norm < 1e-15:
            continue
        ortho.append([x / norm for x in u])
    return ortho
