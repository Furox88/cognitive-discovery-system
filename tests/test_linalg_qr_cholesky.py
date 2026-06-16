from typing import Any, cast

import pytest

from cds.math_utils import cholesky, mat_mul, qr_decomposition, transpose


def _max_err(a: Any, b: Any) -> float:
    return cast(float, max(abs(a[i][j] - b[i][j]) for i in range(len(a)) for j in range(len(a[0]))))


class TestQRDecomposition:
    def test_reconstruction(self) -> None:
        A = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
        Q, R = qr_decomposition(A)
        assert _max_err(mat_mul(Q, R), A) < 1e-9

    def test_q_orthogonal(self) -> None:
        A = [[12.0, -51.0, 4.0], [6.0, 167.0, -68.0], [-4.0, 24.0, -41.0]]
        Q, _ = qr_decomposition(A)
        qtq = mat_mul(transpose(Q), Q)
        n = len(A)
        for i in range(n):
            for j in range(n):
                expect = 1.0 if i == j else 0.0
                assert abs(qtq[i][j] - expect) < 1e-9

    def test_r_upper_triangular(self) -> None:
        A = [[2.0, 1.0], [1.0, 3.0]]
        _, R = qr_decomposition(A)
        assert abs(R[1][0]) < 1e-9

    def test_identity(self) -> None:
        A = [[1.0, 0.0], [0.0, 1.0]]
        Q, R = qr_decomposition(A)
        assert _max_err(mat_mul(Q, R), A) < 1e-12


class TestCholesky:
    def test_known_decomposition(self) -> None:
        M = [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
        L = cholesky(M)
        expected = [[2.0, 0.0, 0.0], [6.0, 1.0, 0.0], [-8.0, 5.0, 3.0]]
        assert _max_err(L, expected) < 1e-9

    def test_reconstruction(self) -> None:
        M = [[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]]
        L = cholesky(M)
        assert _max_err(mat_mul(L, transpose(L)), M) < 1e-9

    def test_lower_triangular(self) -> None:
        M = [[2.0, 1.0], [1.0, 2.0]]
        L = cholesky(M)
        assert L[0][1] == 0.0

    def test_diagonal_matrix(self) -> None:
        M = [[9.0, 0.0], [0.0, 16.0]]
        L = cholesky(M)
        assert abs(L[0][0] - 3.0) < 1e-12
        assert abs(L[1][1] - 4.0) < 1e-12

    def test_not_positive_definite_raises(self) -> None:
        with pytest.raises(ValueError):
            cholesky([[1.0, 2.0], [2.0, 1.0]])

    def test_negative_diagonal_raises(self) -> None:
        with pytest.raises(ValueError):
            cholesky([[-1.0, 0.0], [0.0, -1.0]])
