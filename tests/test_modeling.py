"""Tests for the mathematical modeling module."""

import math

import pytest

from cds.modeling import (
    BinaryOp,
    Constant,
    UnaryFunc,
    Variable,
    differentiate_numerically,
    integrate_numerically,
)


class TestExpressionEvaluation:
    def test_constant(self):
        assert Constant(42.0).evaluate({}) == 42.0

    def test_variable(self):
        assert Variable("x").evaluate({"x": 3.14}) == 3.14

    def test_variable_missing(self):
        with pytest.raises(KeyError):
            Variable("x").evaluate({})

    def test_variable_empty_name(self):
        with pytest.raises(ValueError):
            Variable("")

    def test_binary_add(self):
        expr = BinaryOp("+", Constant(2), Constant(3))
        assert expr.evaluate({}) == 5.0

    def test_binary_sub(self):
        expr = BinaryOp("-", Constant(10), Constant(3))
        assert expr.evaluate({}) == 7.0

    def test_binary_mul(self):
        expr = BinaryOp("*", Variable("x"), Constant(2))
        assert expr.evaluate({"x": 5}) == 10.0

    def test_binary_div(self):
        expr = BinaryOp("/", Constant(10), Constant(4))
        assert expr.evaluate({}) == 2.5

    def test_binary_pow(self):
        expr = BinaryOp("^", Variable("x"), Constant(3))
        assert expr.evaluate({"x": 2}) == 8.0

    def test_unsupported_operator(self):
        with pytest.raises(ValueError):
            BinaryOp("%", Constant(1), Constant(2))

    def test_unary_sin(self):
        expr = UnaryFunc("sin", Constant(0))
        assert expr.evaluate({}) == pytest.approx(0.0)

    def test_unary_cos(self):
        expr = UnaryFunc("cos", Constant(0))
        assert expr.evaluate({}) == pytest.approx(1.0)

    def test_unary_exp(self):
        expr = UnaryFunc("exp", Constant(1))
        assert expr.evaluate({}) == pytest.approx(math.e)

    def test_unary_log(self):
        expr = UnaryFunc("log", Constant(math.e))
        assert expr.evaluate({}) == pytest.approx(1.0)

    def test_unary_sqrt(self):
        expr = UnaryFunc("sqrt", Constant(16))
        assert expr.evaluate({}) == pytest.approx(4.0)

    def test_unary_abs(self):
        expr = UnaryFunc("abs", Constant(-5))
        assert expr.evaluate({}) == 5.0

    def test_unsupported_function(self):
        with pytest.raises(ValueError):
            UnaryFunc("arcsin", Constant(0))

    def test_nested_expression(self):
        # sin(x) + cos(x) at x=0 → 0+1 = 1
        expr = BinaryOp(
            "+",
            UnaryFunc("sin", Variable("x")),
            UnaryFunc("cos", Variable("x")),
        )
        assert expr.evaluate({"x": 0}) == pytest.approx(1.0)


class TestDifferentiation:
    def test_derivative_of_x_squared(self):
        # d/dx(x^2) = 2x; at x=3 → 6
        expr = BinaryOp("^", Variable("x"), Constant(2))
        result = differentiate_numerically(expr, "x", {"x": 3.0})
        assert result == pytest.approx(6.0, abs=1e-5)

    def test_derivative_of_sin(self):
        # d/dx(sin(x)) = cos(x); at x=0 → 1
        expr = UnaryFunc("sin", Variable("x"))
        result = differentiate_numerically(expr, "x", {"x": 0.0})
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_variable_not_in_env(self):
        expr = Variable("x")
        with pytest.raises(KeyError):
            differentiate_numerically(expr, "y", {"x": 1.0})

    def test_negative_step_rejected(self):
        expr = Variable("x")
        with pytest.raises(ValueError):
            differentiate_numerically(expr, "x", {"x": 1.0}, h=-1e-8)


class TestIntegration:
    def test_integral_of_x(self):
        # ∫₀¹ x dx = 0.5
        expr = Variable("x")
        result = integrate_numerically(expr, "x", 0, 1, {})
        assert result == pytest.approx(0.5, abs=1e-4)

    def test_integral_of_x_squared(self):
        # ∫₀¹ x² dx = 1/3
        expr = BinaryOp("^", Variable("x"), Constant(2))
        result = integrate_numerically(expr, "x", 0, 1, {})
        assert result == pytest.approx(1 / 3, abs=1e-4)

    def test_integral_of_sin(self):
        # ∫₀^π sin(x) dx = 2
        expr = UnaryFunc("sin", Variable("x"))
        result = integrate_numerically(expr, "x", 0, math.pi, {})
        assert result == pytest.approx(2.0, abs=1e-3)

    def test_invalid_n(self):
        expr = Constant(1)
        with pytest.raises(ValueError):
            integrate_numerically(expr, "x", 0, 1, {}, n=0)

    def test_invalid_bounds(self):
        expr = Constant(1)
        with pytest.raises(ValueError):
            integrate_numerically(expr, "x", 5, 1, {})
