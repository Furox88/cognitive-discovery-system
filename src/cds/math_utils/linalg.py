"""Basic linear algebra ops — pure python, no deps."""
from __future__ import annotations

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
    # cofactor expansion along first row
    det = 0.0
    for j in range(n):
        minor = [row[:j] + row[j + 1:] for row in m[1:]]
        det += ((-1) ** j) * m[0][j] * determinant(minor)
    return det
