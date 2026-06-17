"""Tests for math_utils module."""

import math

from cds.math_utils.calculus import derivative, gradient, integral
from cds.math_utils.linalg import determinant, dot, mat_mul, transpose


def test_derivative_of_x_squared() -> None:
    # d/dx(x^2) = 2x, at x=3 should be 6
    result = derivative(lambda x: x**2, 3.0)
    assert abs(result - 6.0) < 1e-5


def test_derivative_of_sin() -> None:
    # d/dx(sin(x)) = cos(x), at x=0 should be 1
    result = derivative(math.sin, 0.0)
    assert abs(result - 1.0) < 1e-5


def test_integral_of_x_squared() -> None:
    # integral of x^2 from 0 to 3 = 9
    result = integral(lambda x: x**2, 0, 3)
    assert abs(result - 9.0) < 0.01


def test_integral_of_sin() -> None:
    # integral of sin(x) from 0 to pi = 2
    result = integral(math.sin, 0, math.pi)
    assert abs(result - 2.0) < 0.001


def test_gradient() -> None:
    def f(x: float, y: float) -> float:
        return x**2 + y**2

    g = gradient(f, [3.0, 4.0])
    assert abs(g[0] - 6.0) < 1e-4
    assert abs(g[1] - 8.0) < 1e-4


def test_dot_product() -> None:
    assert dot([1, 2, 3], [4, 5, 6]) == 32


def test_mat_mul() -> None:
    a = [[1.0, 2.0], [3.0, 4.0]]
    b = [[5.0, 6.0], [7.0, 8.0]]
    result = mat_mul(a, b)
    assert result == [[19.0, 22.0], [43.0, 50.0]]


def test_transpose() -> None:
    m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    assert transpose(m) == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]


def test_determinant_2x2() -> None:
    m = [[1.0, 2.0], [3.0, 4.0]]
    assert abs(determinant(m) - (-2)) < 1e-9


def test_determinant_3x3() -> None:
    m = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]
    assert abs(determinant(m) - 1.0) < 1e-9


def test_derivative_of_exp() -> None:
    # d/dx(e^x) = e^x, at x=1
    result = derivative(math.exp, 1.0)
    assert abs(result - math.e) < 1e-4


def test_integral_constant() -> None:
    # integral of 5 from 0 to 3 = 15
    result = integral(lambda x: 5.0, 0, 3)
    assert abs(result - 15.0) < 0.01


def test_determinant_identity() -> None:
    m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert abs(determinant(m) - 1.0) < 1e-9


def test_transpose_square() -> None:
    m = [[1.0, 2.0], [3.0, 4.0]]
    assert transpose(m) == [[1.0, 3.0], [2.0, 4.0]]


def test_dot_orthogonal() -> None:
    assert dot([1, 0, 0], [0, 1, 0]) == 0


def test_mat_mul_identity() -> None:
    I = [[1.0, 0.0], [0.0, 1.0]]  # noqa: E741
    a = [[5.0, 6.0], [7.0, 8.0]]
    assert mat_mul(I, a) == a


def test_determinant_singular() -> None:
    m = [[1.0, 2.0], [2.0, 4.0]]
    assert abs(determinant(m)) < 1e-9


def test_integral_sin_half_period() -> None:
    result = integral(math.sin, 0, math.pi)
    assert abs(result - 2.0) < 1e-6


def test_derivative_of_constant() -> None:
    result = derivative(lambda x: 5.0, 3.0)
    assert abs(result) < 1e-6


def test_transpose_rect() -> None:
    m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    t = transpose(m)
    assert len(t) == 3
    assert len(t[0]) == 2
    assert t[0] == [1.0, 4.0]


def test_determinant_3x3_unit() -> None:
    m = [[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]]
    d = determinant(m)
    assert abs(d - 1.0) < 1e-9


def test_dot_parallel() -> None:
    assert dot([2, 0], [3, 0]) == 6
