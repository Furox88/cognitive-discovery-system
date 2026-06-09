"""Tests for math_utils module."""
import math

from cds.math_utils.calculus import derivative, gradient, integral
from cds.math_utils.linalg import determinant, dot, mat_mul, transpose


def test_derivative_of_x_squared():
    # d/dx(x^2) = 2x, at x=3 should be 6
    result = derivative(lambda x: x ** 2, 3.0)
    assert abs(result - 6.0) < 1e-5


def test_derivative_of_sin():
    # d/dx(sin(x)) = cos(x), at x=0 should be 1
    result = derivative(math.sin, 0.0)
    assert abs(result - 1.0) < 1e-5


def test_integral_of_x_squared():
    # integral of x^2 from 0 to 3 = 9
    result = integral(lambda x: x ** 2, 0, 3)
    assert abs(result - 9.0) < 0.01


def test_integral_of_sin():
    # integral of sin(x) from 0 to pi = 2
    result = integral(math.sin, 0, math.pi)
    assert abs(result - 2.0) < 0.001


def test_gradient():
    def f(x: float, y: float) -> float:
        return x ** 2 + y ** 2
    g = gradient(f, [3.0, 4.0])
    assert abs(g[0] - 6.0) < 1e-4
    assert abs(g[1] - 8.0) < 1e-4


def test_dot_product():
    assert dot([1, 2, 3], [4, 5, 6]) == 32


def test_mat_mul():
    a = [[1, 2], [3, 4]]
    b = [[5, 6], [7, 8]]
    result = mat_mul(a, b)
    assert result == [[19, 22], [43, 50]]


def test_transpose():
    m = [[1, 2, 3], [4, 5, 6]]
    assert transpose(m) == [[1, 4], [2, 5], [3, 6]]


def test_determinant_2x2():
    m = [[1, 2], [3, 4]]
    assert abs(determinant(m) - (-2)) < 1e-9


def test_determinant_3x3():
    m = [[1, 2, 3], [0, 1, 4], [5, 6, 0]]
    assert abs(determinant(m) - 1.0) < 1e-9
