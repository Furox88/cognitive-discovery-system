"""Tests for extended linear algebra functions."""

import math

import pytest

from cds.math_utils.linalg import (
    dot,
    gram_schmidt,
    identity,
    lu_decomposition,
    mat_mul,
    matrix_inverse,
    power_iteration,
    solve_linear,
)


class TestIdentity:
    def test_2x2(self) -> None:
        ident = identity(2)
        assert ident == [[1, 0], [0, 1]]

    def test_3x3(self) -> None:
        ident = identity(3)
        for i in range(3):
            for j in range(3):
                assert ident[i][j] == (1.0 if i == j else 0.0)


class TestLUDecomposition:
    def test_2x2(self) -> None:
        A = [[2.0, 1.0], [4.0, 3.0]]
        P, L, U = lu_decomposition(A)
        # verify P_inv * L * U = A. Since P is symmetric orthogonal here, P = P_inv
        from cds.math_utils.linalg import transpose

        P_inv = transpose(P)
        product = mat_mul(P_inv, mat_mul(L, U))
        for i in range(2):
            for j in range(2):
                assert abs(product[i][j] - A[i][j]) < 1e-10

    def test_3x3(self) -> None:
        A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]]
        P, L, U = lu_decomposition(A)
        from cds.math_utils.linalg import transpose

        P_inv = transpose(P)
        product = mat_mul(P_inv, mat_mul(L, U))
        for i in range(3):
            for j in range(3):
                assert abs(product[i][j] - A[i][j]) < 1e-10

    def test_singular_raises(self) -> None:
        A = [[0.0, 1.0], [0.0, 1.0]]
        with pytest.raises(ValueError):
            lu_decomposition(A)

    def test_L_is_lower_triangular(self) -> None:
        A = [[2.0, 1.0], [4.0, 3.0]]
        P, L, U = lu_decomposition(A)
        assert L[0][1] == 0.0
        assert L[0][0] == 1.0
        assert L[1][1] == 1.0

    def test_small_scale_well_conditioned_factors(self) -> None:
        # A tiny but well-conditioned matrix (all entries ~1e-20). With the
        # old *absolute* pivot threshold (NEAR_ZERO = 1e-15) every pivot fell
        # below it and the matrix was wrongly rejected as singular; the
        # scale-relative threshold now recognizes it as well-conditioned.
        A = [[2.0e-20, 1.0e-20], [1.0e-20, 3.0e-20]]
        P, L, U = lu_decomposition(A)
        from cds.math_utils.linalg import transpose

        P_inv = transpose(P)
        product = mat_mul(P_inv, mat_mul(L, U))
        for i in range(2):
            for j in range(2):
                assert abs(product[i][j] - A[i][j]) < 1e-29

    def test_small_scale_solve_recovers_solution(self) -> None:
        # End-to-end check that the scale-relative threshold lets a real
        # small-magnitude system be solved accurately.
        A = [[2.0e-20, 1.0e-20], [1.0e-20, 3.0e-20]]
        b = [5.0e-20, 8.0e-20]
        x = solve_linear(A, b)
        # A x should reproduce b
        for i in range(2):
            val = sum(A[i][j] * x[j] for j in range(2))
            assert abs(val - b[i]) < 1e-29

    def test_genuinely_singular_still_rejected_at_small_scale(self) -> None:
        # Regression guard: the relative threshold must NOT rubber-stamp a
        # truly rank-deficient matrix just because its entries are small.
        # Second row is a multiple of the first -> singular at any scale.
        A = [[1.0e-20, 2.0e-20], [2.0e-20, 4.0e-20]]
        with pytest.raises(ValueError):
            lu_decomposition(A)


class TestSolveLinear:
    def test_simple_system(self) -> None:
        # 2x + y = 5, 4x + 3y = 11  =>  x=2, y=1
        A = [[2.0, 1.0], [4.0, 3.0]]
        b = [5.0, 11.0]
        x = solve_linear(A, b)
        assert abs(x[0] - 2.0) < 1e-10
        assert abs(x[1] - 1.0) < 1e-10

    def test_3x3_system(self) -> None:
        A = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]
        b = [14.0, 13.0, 11.0]
        x = solve_linear(A, b)
        # verify Ax = b
        for i in range(3):
            val = sum(A[i][j] * x[j] for j in range(3))
            assert abs(val - b[i]) < 1e-10


class TestMatrixInverse:
    def test_2x2_inverse(self) -> None:
        A = [[4.0, 7.0], [2.0, 6.0]]
        inv = matrix_inverse(A)
        product = mat_mul(A, inv)
        ident = identity(2)
        for i in range(2):
            for j in range(2):
                assert abs(product[i][j] - ident[i][j]) < 1e-10

    def test_3x3_inverse(self) -> None:
        A = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]
        inv = matrix_inverse(A)
        product = mat_mul(A, inv)
        ident = identity(3)
        for i in range(3):
            for j in range(3):
                assert abs(product[i][j] - ident[i][j]) < 1e-10


class TestPowerIteration:
    def test_dominant_eigenvalue(self) -> None:
        # [[2, 1], [1, 2]] has eigenvalues 3 and 1
        A = [[2.0, 1.0], [1.0, 2.0]]
        eigenvalue, eigenvector = power_iteration(A)
        assert abs(eigenvalue - 3.0) < 1e-6

    def test_eigenvector_normalized(self) -> None:
        A = [[2.0, 1.0], [1.0, 2.0]]
        _, v = power_iteration(A)
        norm = math.sqrt(sum(x**2 for x in v))
        assert abs(norm - 1.0) < 1e-10

    def test_diagonal_matrix(self) -> None:
        A = [[5.0, 0.0], [0.0, 2.0]]
        eigenvalue, _ = power_iteration(A)
        assert abs(eigenvalue - 5.0) < 1e-6


class TestGramSchmidt:
    def test_orthonormal_output(self) -> None:
        vectors = [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
        ortho = gram_schmidt(vectors)
        assert len(ortho) == 3
        # check orthonormality
        for i in range(3):
            assert abs(dot(ortho[i], ortho[i]) - 1.0) < 1e-10
            for j in range(i + 1, 3):
                assert abs(dot(ortho[i], ortho[j])) < 1e-10

    def test_2d(self) -> None:
        vectors = [[3.0, 0.0], [1.0, 1.0]]
        ortho = gram_schmidt(vectors)
        assert abs(dot(ortho[0], ortho[1])) < 1e-10

    def test_dependent_vectors_skipped(self) -> None:
        vectors = [[1.0, 0.0], [2.0, 0.0]]
        ortho = gram_schmidt(vectors)
        assert len(ortho) == 1
