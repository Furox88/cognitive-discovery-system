"""Tests for optimization module."""
import math

from cds.optimization.minimize import (
    OptResult,
    adam,
    gradient_descent,
    line_search,
    newton_method,
)

# --- gradient descent ---

def test_gd_quadratic():
    # minimize x^2, minimum at x=0
    result = gradient_descent(lambda x: x ** 2, x0=5.0, lr=0.1)
    assert result.converged
    assert abs(result.x) < 1e-4


def test_gd_shifted_quadratic():
    # minimize (x-3)^2, minimum at x=3
    result = gradient_descent(lambda x: (x - 3) ** 2, x0=0.0, lr=0.1)
    assert result.converged
    assert abs(result.x - 3) < 1e-4


def test_gd_returns_opt_result():
    result = gradient_descent(lambda x: x ** 2, x0=1.0)
    assert isinstance(result, OptResult)
    assert result.iterations >= 0


def test_gd_cos_local_min():
    # cos(x) near x=pi has minimum
    result = gradient_descent(math.cos, x0=2.5, lr=0.1)
    assert abs(result.x - math.pi) < 0.1


# --- Newton's method ---

def test_newton_sqrt_2():
    # find root of x^2 - 2 = 0 => x = sqrt(2)
    result = newton_method(lambda x: x ** 2 - 2, x0=1.5)
    assert result.converged
    assert abs(result.x - math.sqrt(2)) < 1e-8


def test_newton_sin():
    # root of sin(x) near pi
    result = newton_method(math.sin, x0=3.0)
    assert result.converged
    assert abs(result.x - math.pi) < 1e-8


def test_newton_cubic():
    # x^3 - 8 = 0 => x = 2
    result = newton_method(lambda x: x ** 3 - 8, x0=3.0)
    assert result.converged
    assert abs(result.x - 2.0) < 1e-6


# --- Adam optimizer ---

def test_adam_quadratic():
    result = adam(lambda x: x ** 2, x0=5.0, lr=0.1)
    assert result.converged
    assert abs(result.x) < 0.1


def test_adam_shifted():
    result = adam(lambda x: (x - 2) ** 2, x0=10.0, lr=0.1)
    assert abs(result.x - 2) < 0.5


# --- Vector (Multi-dimensional) Support ---

def test_gd_vector_quadratic():
    # minimize f(x, y) = x^2 + y^2, minimum at [0, 0]
    result = gradient_descent(lambda v: v[0]**2 + v[1]**2, x0=[5.0, -3.0], lr=0.1)
    assert result.converged
    assert abs(result.x[0]) < 1e-4
    assert abs(result.x[1]) < 1e-4

def test_gd_vector_rosenbrock_approximation():
    # simple 2d parabola shifted
    result = gradient_descent(lambda v: (v[0]-1)**2 + (v[1]-2)**2, x0=[0.0, 0.0], lr=0.1)
    assert result.converged
    assert abs(result.x[0] - 1.0) < 1e-4
    assert abs(result.x[1] - 2.0) < 1e-4

def test_adam_vector_quadratic():
    result = adam(lambda v: v[0]**2 + v[1]**2, x0=[5.0, -5.0], lr=0.1)
    assert result.converged
    assert abs(result.x[0]) < 0.1
    assert abs(result.x[1]) < 0.1


# --- Golden section search ---

def test_line_search_quadratic():
    result = line_search(lambda x: (x - 2) ** 2, a=0, b=5)
    assert result.converged
    assert abs(result.x - 2) < 1e-6


def test_line_search_sin():
    # sin(x) minimum in [pi/2, 3pi/2] is at 3pi/2
    result = line_search(math.sin, a=math.pi / 2, b=3 * math.pi / 2)
    assert result.converged
    assert abs(result.x - 3 * math.pi / 2) < 1e-4


def test_line_search_abs():
    result = line_search(abs, a=-3, b=5)
    assert result.converged
    assert abs(result.x) < 1e-5
