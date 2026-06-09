import math
import pytest
from cds.modeling import (
    BinaryOp, Constant, UnaryFunc, Variable,
    differentiate_numerically, integrate_numerically,
)


class TestExpressions:
    def test_constant(self):
        assert Constant(42.0).evaluate({}) == 42.0

    def test_variable(self):
        assert Variable("x").evaluate({"x": 3.14}) == 3.14

    def test_variable_missing(self):
        with pytest.raises(KeyError):
            Variable("x").evaluate({})

    def test_variable_empty(self):
        with pytest.raises(ValueError):
            Variable("")

    def test_add(self):
        assert BinaryOp("+", Constant(2), Constant(3)).evaluate({}) == 5.0

    def test_sub(self):
        assert BinaryOp("-", Constant(10), Constant(3)).evaluate({}) == 7.0

    def test_mul(self):
        assert BinaryOp("*", Variable("x"), Constant(2)).evaluate({"x": 5}) == 10.0

    def test_div(self):
        assert BinaryOp("/", Constant(10), Constant(4)).evaluate({}) == 2.5

    def test_pow(self):
        assert BinaryOp("^", Variable("x"), Constant(3)).evaluate({"x": 2}) == 8.0

    def test_bad_op(self):
        with pytest.raises(ValueError):
            BinaryOp("%", Constant(1), Constant(2))

    def test_sin(self):
        assert UnaryFunc("sin", Constant(0)).evaluate({}) == pytest.approx(0.0)

    def test_cos(self):
        assert UnaryFunc("cos", Constant(0)).evaluate({}) == pytest.approx(1.0)

    def test_exp(self):
        assert UnaryFunc("exp", Constant(1)).evaluate({}) == pytest.approx(math.e)

    def test_log(self):
        assert UnaryFunc("log", Constant(math.e)).evaluate({}) == pytest.approx(1.0)

    def test_sqrt(self):
        assert UnaryFunc("sqrt", Constant(16)).evaluate({}) == pytest.approx(4.0)

    def test_abs(self):
        assert UnaryFunc("abs", Constant(-5)).evaluate({}) == 5.0

    def test_bad_func(self):
        with pytest.raises(ValueError):
            UnaryFunc("arcsin", Constant(0))

    def test_nested(self):
        # sin(x) + cos(x) at x=0 -> 1
        expr = BinaryOp("+", UnaryFunc("sin", Variable("x")), UnaryFunc("cos", Variable("x")))
        assert expr.evaluate({"x": 0}) == pytest.approx(1.0)


class TestDifferentiation:
    def test_x_squared(self):
        # d/dx(x^2) = 2x -> 6 at x=3
        expr = BinaryOp("^", Variable("x"), Constant(2))
        assert differentiate_numerically(expr, "x", {"x": 3.0}) == pytest.approx(6.0, abs=1e-5)

    def test_sin(self):
        # d/dx(sin(x)) = cos(x) -> 1 at x=0
        expr = UnaryFunc("sin", Variable("x"))
        assert differentiate_numerically(expr, "x", {"x": 0.0}) == pytest.approx(1.0, abs=1e-5)

    def test_missing_var(self):
        with pytest.raises(KeyError):
            differentiate_numerically(Variable("x"), "y", {"x": 1.0})

    def test_bad_step(self):
        with pytest.raises(ValueError):
            differentiate_numerically(Variable("x"), "x", {"x": 1.0}, h=-1e-8)


class TestIntegration:
    def test_x(self):
        # integral of x from 0 to 1 = 0.5
        assert integrate_numerically(Variable("x"), "x", 0, 1, {}) == pytest.approx(0.5, abs=1e-4)

    def test_x_squared(self):
        # integral of x^2 from 0 to 1 = 1/3
        expr = BinaryOp("^", Variable("x"), Constant(2))
        assert integrate_numerically(expr, "x", 0, 1, {}) == pytest.approx(1/3, abs=1e-4)

    def test_sin(self):
        # integral of sin(x) from 0 to pi = 2
        expr = UnaryFunc("sin", Variable("x"))
        assert integrate_numerically(expr, "x", 0, math.pi, {}) == pytest.approx(2.0, abs=1e-3)

    def test_bad_n(self):
        with pytest.raises(ValueError):
            integrate_numerically(Constant(1), "x", 0, 1, {}, n=0)

    def test_bad_bounds(self):
        with pytest.raises(ValueError):
            integrate_numerically(Constant(1), "x", 5, 1, {})
