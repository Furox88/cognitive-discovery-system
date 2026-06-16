"""Linear algebra — pure Python, no dependencies.

References:
    - Golub, G.H. & Van Loan, C.F. Matrix Computations (4th ed.)
    - Trefethen, L.N. & Bau, D. Numerical Linear Algebra.
    - Von Mises, R. (1929). Praktische Verfahren der Gleichungsauflösung (power iteration).
    - Householder, A.S. (1958). Unitary triangularization of a nonsymmetric
      matrix. JACM 5(4), 339-342.
    - Cholesky, A.-L. / Benoît (1924). Méthode de résolution des équations
      normales (Bulletin Géodésique, 2, 67-77).
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
    """Matrix multiplication A * B (Optimized Pure Python).
    
    Uses pre-transposition to exploit row-major access patterns,
    making it significantly faster by using Python's internal zip/sum.
    """
    rows_a, cols_a = len(a), len(a[0])
    rows_b, cols_b = len(b), len(b[0])
    if cols_a != rows_b:
        raise ValueError(f"incompatible shapes: {rows_a}x{cols_a} and {rows_b}x{cols_b}")
    
    # Pre-transpose B to make column access O(1) row access
    # This is an 'Intelligence' boost: treating memory access patterns as a priority
    b_T = list(zip(*b))
    
    return [[sum(ai * bi for ai, bi in zip(row_a, col_b)) 
             for col_b in b_T] 
            for row_a in a]


def transpose(m: Matrix) -> Matrix:
    if not m:
        return []
    return [[m[i][j] for i in range(len(m))] for j in range(len(m[0]))]


def determinant(m: Matrix) -> float:
    """Compute matrix determinant using PLU decomposition (O(N^3)).

    Avoids the O(N!) complexity of minor expansion.
    """
    n = len(m)
    if n == 0:
        return 1.0
    if n == 1:
        return m[0][0]

    try:
        P, L, U = lu_decomposition(m)
    except ValueError:
        # If matrix is singular, determinant is 0
        return 0.0

    # Determinant of LU is product of diag(U) 
    # (diag(L) is all 1s).
    det = 1.0
    for i in range(n):
        det *= U[i][i]

    # Determinant of P is (-1)^s where s is number of row swaps.
    # We compute it using cycle decomposition: s = n - number_of_cycles.
    num_cycles = 0
    p_indices = [row.index(1.0) for row in P]
    visited = [False] * n
    for i in range(n):
        if not visited[i]:
            num_cycles += 1
            curr = i
            while not visited[curr]:
                visited[curr] = True
                curr = p_indices[curr]
            
    return float(det * ((-1) ** (n - num_cycles)))


def identity(n: int) -> Matrix:
    """Create n×n identity matrix."""
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def lu_decomposition(m: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """LU decomposition with partial pivoting (PA = LU).

    A = P_inv * L * U where P_inv is a permutation matrix,
    L is lower triangular (ones on diagonal) and U is upper triangular.

    Returns:
        P, L, U matrices.

    Raises:
        ValueError: if matrix is singular
    """
    n = len(m)
    P = identity(n)
    L = [[0.0] * n for _ in range(n)]
    U = [row[:] for row in m]

    for k in range(n):
        # Partial pivoting
        pivot_idx = k
        max_val = abs(U[k][k])
        for i in range(k + 1, n):
            if abs(U[i][k]) > max_val:
                max_val = abs(U[i][k])
                pivot_idx = i

        if max_val < 1e-15:
            raise ValueError("zero pivot encountered — matrix may be singular")

        if pivot_idx != k:
            U[k], U[pivot_idx] = U[pivot_idx], U[k]
            P[k], P[pivot_idx] = P[pivot_idx], P[k]
            L[k], L[pivot_idx] = L[pivot_idx], L[k]

        L[k][k] = 1.0
        for i in range(k + 1, n):
            factor = U[i][k] / U[k][k]
            L[i][k] = factor
            for j in range(k, n):
                U[i][j] -= factor * U[k][j]

    return P, L, U


def solve_linear(A: Matrix, b: Vector) -> Vector:
    """Solve Ax = b using PLU decomposition.

    Solves LUx = Pb.

    Raises:
        ValueError: if matrix is singular
    """
    n = len(A)
    P, L, U = lu_decomposition(A)

    # Apply permutation: Pb
    pb = [sum(P[i][j] * b[j] for j in range(n)) for i in range(n)]

    # forward: Ly = Pb
    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(L[i][j] * y[j] for j in range(i))

    # backward: Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < 1e-15:
            raise ValueError("singular matrix")
        x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

    return x


def matrix_inverse(m: Matrix) -> Matrix:
    """Compute matrix inverse via PLU decomposition.

    Reuses a single P, L, U factorization and solves A * x_i = e_i
    for each column of the identity matrix to build the inverse.

    Raises:
        ValueError: if matrix is singular
    """
    n = len(m)
    P, L, U = lu_decomposition(m)
    inv = [[0.0] * n for _ in range(n)]

    for col in range(n):
        # e is the standard basis vector
        b = [0.0] * n
        b[col] = 1.0

        # Apply permutation: Pb
        pb = [sum(P[i][j] * b[j] for j in range(n)) for i in range(n)]

        # forward: Ly = Pb
        y = [0.0] * n
        for i in range(n):
            y[i] = pb[i] - sum(L[i][j] * y[j] for j in range(i))

        # backward: Ux = y
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            if abs(U[i][i]) < 1e-15:
                raise ValueError("singular matrix")
            x[i] = (y[i] - sum(U[i][j] * x[j] for j in range(i + 1, n))) / U[i][i]

        for row in range(n):
            inv[row][col] = x[row]
            
    return inv


def power_iteration(
    m: Matrix, max_iter: int = 1000, tol: float = 1e-10,
) -> tuple[float, Vector]:
    """Find dominant eigenvalue and eigenvector using power iteration.

    Von Mises iteration (1929). Optimized with scaling to prevent overflow.

    Args:
        m: square matrix
        max_iter: iteration limit
        tol: convergence tolerance

    Returns:
        (eigenvalue, eigenvector) tuple
    """
    n = len(m)
    v = [1.0] * n
    
    # Initial scaling
    max_val = max(abs(x) for x in v)
    v = [x / max_val for x in v]

    eigenvalue = 0.0
    for _ in range(max_iter):
        # w = A * v
        w = [sum(m[i][j] * v[j] for j in range(n)) for i in range(n)]
        
        # Scaling to prevent overflow in large systems
        # norm = sqrt(sum(w_i^2)). CPython floats overflow to inf rather
        # than raising OverflowError, so we detect both cases and fall back
        # to absolute-max scaling, which is safe for any magnitude.
        squared_sum = sum(x * x for x in w)
        if math.isinf(squared_sum):
            norm = max(abs(x) for x in w)
        else:
            try:
                norm = math.sqrt(squared_sum)
            except OverflowError:  # pragma: no cover - defensive for non-CPython libm
                # Defensive: still raised on some platforms for subnormal inputs
                norm = max(abs(x) for x in w)

        if norm < 1e-15:
            break
            
        v_new = [x / norm for x in w]
        
        # Rayleigh quotient: (v^T * A * v) / (v^T * v)
        # Accurate for any normalization (L2 or L-inf)
        numerator = sum(v_new[i] * sum(m[i][j] * v_new[j] for j in range(n)) for i in range(n))
        denominator = sum(vi * vi for vi in v_new)
        new_eigenvalue = numerator / denominator if denominator > 1e-15 else 0.0

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


def qr_decomposition(m: Matrix) -> tuple[Matrix, Matrix]:
    """QR decomposition via Householder reflections.

    Factorizes A (n×n) into an orthogonal matrix Q and upper-triangular R
    such that A = Q R. Householder triangularization is backward stable and
    preferred over classical Gram-Schmidt for numerical work.

    Reference:
        Householder, A. S. (1958). "Unitary triangularization of a
        nonsymmetric matrix." Journal of the ACM, 5(4), 339-342.
        See also Golub & Van Loan, §5.2; Trefethen & Bau, Lecture 10.

    Args:
        m: square matrix

    Returns:
        (Q, R) with Q orthogonal and R upper triangular
    """
    n = len(m)
    R = [row[:] for row in m]
    Q = identity(n)

    for k in range(n - 1):
        # column vector x = R[k:, k]
        x = [R[i][k] for i in range(k, n)]
        norm_x = math.sqrt(sum(xi * xi for xi in x))
        if norm_x < 1e-15:
            continue
        # Householder vector v
        alpha = -norm_x if x[0] >= 0 else norm_x
        v = x[:]
        v[0] -= alpha
        norm_v = math.sqrt(sum(vi * vi for vi in v))
        if norm_v < 1e-15:  # pragma: no cover - unreachable: norm_x>0 implies norm_v>0
            continue
        v = [vi / norm_v for vi in v]

        # apply H = I - 2 v v^T to R (rows k..n-1)
        for j in range(n):
            dot_vr = sum(v[i] * R[k + i][j] for i in range(n - k))
            for i in range(n - k):
                R[k + i][j] -= 2.0 * v[i] * dot_vr

        # accumulate Q = Q H (columns k..n-1)
        for i in range(n):
            dot_qv = sum(Q[i][k + j] * v[j] for j in range(n - k))
            for j in range(n - k):
                Q[i][k + j] -= 2.0 * dot_qv * v[j]

    return Q, R


def cholesky(m: Matrix) -> Matrix:
    """Cholesky decomposition of a symmetric positive-definite matrix.

    Returns the lower-triangular L such that A = L L^T. Roughly twice as
    efficient as LU for SPD systems and numerically stable.

    Reference:
        Benoît, C. (1924). "Note sur une méthode de résolution des équations
        normales... (Procédé du Commandant Cholesky)." Bulletin Géodésique,
        2, 67-77. See also Golub & Van Loan, §4.2.

    Args:
        m: symmetric positive-definite matrix

    Returns:
        lower-triangular matrix L with A = L L^T

    Raises:
        ValueError: if the matrix is not positive definite
    """
    n = len(m)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = sum(L[i][k] * L[j][k] for k in range(j))
            if i == j:
                diag = m[i][i] - s
                if diag <= 0.0:
                    raise ValueError("matrix is not positive definite")
                L[i][j] = math.sqrt(diag)
            else:
                L[i][j] = (m[i][j] - s) / L[j][j]
    return L
