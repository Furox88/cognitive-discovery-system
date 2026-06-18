"""Tests for :mod:`cds.nlp.autograd`.

Covers the core scalar autograd engine: forward + backward for each
op, gradient correctness via numerical comparison, the topological
traversal, the operator overloading sugar, and the ``no_grad``
context manager.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pytest

from cds.nlp.autograd import (
    Parameter,
    Tensor,
    add,
    div,
    exp,
    log,
    matmul,
    mul,
    neg,
    no_grad,
    relu,
    sub,
)

# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #


def numerical_grad(
    fn: Callable[[float], float],
    x: float,
    eps: float = 1e-5,
) -> float:
    """Symmetric numerical derivative of a scalar function."""
    return (fn(x + eps) - fn(x - eps)) / (2.0 * eps)


# ---------------------------------------------------------------------- #
# Tensor basics
# ---------------------------------------------------------------------- #


class TestTensor:
    """Construction, repr, grad initialisation."""

    def test_construction(self) -> None:
        t = Tensor(data=1.5, requires_grad=True)
        assert t.data == 1.5
        assert t.requires_grad is True
        assert t.grad == 0.0

    def test_no_grad_by_default(self) -> None:
        t = Tensor(data=1.5)
        assert t.requires_grad is False

    def test_repr_contains_data(self) -> None:
        t = Tensor(data=2.0, requires_grad=True)
        assert "2.0" in repr(t)

    def test_repr_omits_grad_when_disabled(self) -> None:
        t = Tensor(data=2.0, requires_grad=False)
        assert "grad" not in repr(t)

    def test_backward_without_grad_raises(self) -> None:
        t = Tensor(data=1.0, requires_grad=False)
        with pytest.raises(RuntimeError, match="requires_grad"):
            t.backward()

    def test_zero_grad(self) -> None:
        t = Tensor(data=2.0, requires_grad=True)
        t.grad = 5.0
        t.zero_grad()
        assert t.grad == 0.0

    def test_neg_operator(self) -> None:
        """Unary ``-Tensor`` dispatches to :func:`neg` and propagates grad."""
        a = Tensor(data=5.0, requires_grad=True)
        c = -a  # __neg__
        assert c.data == -5.0
        c.backward()
        assert a.grad == -1.0

    def test_pos_operator_is_identity(self) -> None:
        """Unary ``+Tensor`` returns the node unchanged."""
        a = Tensor(data=1.25, requires_grad=True)
        assert (+a) is a

    def test_zero_grad_diamond_graph(self) -> None:
        """``zero_grad`` walks a non-trivial graph with a shared parent
        (the diamond ``a -> b, a -> c, (b,c) -> d``). The shared node ``a``
        is reachable via two paths, so its ``visited`` short-circuit
        (``continue``) and the unvisited-child push both fire.
        """
        a = Tensor(data=2.0, requires_grad=True)
        b = a + Tensor(data=1.0, requires_grad=True)
        c = a * Tensor(data=3.0, requires_grad=True)
        d = b + c
        d.backward()
        assert a.grad != 0.0
        # Reset across the whole reachable subgraph.
        d.zero_grad()
        for node in (a, b, c, d):
            assert node.grad == 0.0

    def test_binop_rejects_unsupported_type(self) -> None:
        """An operand that is neither Tensor nor number raises TypeError."""
        a = Tensor(data=1.0, requires_grad=True)
        with pytest.raises(TypeError, match="unsupported operand type"):
            _ = a + "not_a_number"  # type: ignore[operator]


# ---------------------------------------------------------------------- #
# Addition / Subtraction
# ---------------------------------------------------------------------- #


class TestAddSub:
    def test_add_forward(self) -> None:
        c = add(Tensor(data=2.0, requires_grad=True), Tensor(data=3.0, requires_grad=True))
        assert c.data == 5.0

    def test_add_grad(self) -> None:
        a = Tensor(data=2.0, requires_grad=True)
        b = Tensor(data=3.0, requires_grad=True)
        c = add(a, b)
        c.backward()
        assert a.grad == 1.0
        assert b.grad == 1.0

    def test_sub_grad(self) -> None:
        a = Tensor(data=5.0, requires_grad=True)
        b = Tensor(data=2.0, requires_grad=True)
        c = sub(a, b)
        c.backward()
        assert a.grad == 1.0
        assert b.grad == -1.0

    def test_operator_overload(self) -> None:
        a = Tensor(data=2.0, requires_grad=True)
        b = Tensor(data=3.0, requires_grad=True)
        c = a + b
        assert c.data == 5.0
        c.backward()
        assert a.grad == 1.0


# ---------------------------------------------------------------------- #
# Multiplication / Division
# ---------------------------------------------------------------------- #


class TestMulDiv:
    def test_mul_grad(self) -> None:
        a = Tensor(data=3.0, requires_grad=True)
        b = Tensor(data=4.0, requires_grad=True)
        c = mul(a, b)
        c.backward()
        assert a.grad == 4.0  # dc/da = b
        assert b.grad == 3.0  # dc/db = a

    def test_div_grad(self) -> None:
        a = Tensor(data=6.0, requires_grad=True)
        b = Tensor(data=2.0, requires_grad=True)
        c = div(a, b)
        c.backward()
        assert abs(a.grad - 0.5) < 1e-12  # dc/da = 1/b
        assert abs(b.grad - (-1.5)) < 1e-12  # dc/db = -a/b^2

    def test_neg_grad(self) -> None:
        a = Tensor(data=5.0, requires_grad=True)
        c = neg(a)
        c.backward()
        assert a.grad == -1.0


# ---------------------------------------------------------------------- #
# Exponentials / Logs
# ---------------------------------------------------------------------- #


class TestExpLog:
    def test_exp_grad(self) -> None:
        a = Tensor(data=2.0, requires_grad=True)
        c = exp(a)
        c.backward()
        # dc/da = exp(a) * 1
        assert abs(a.grad - math.exp(2.0)) < 1e-12

    def test_log_grad(self) -> None:
        a = Tensor(data=2.0, requires_grad=True)
        c = log(a)
        c.backward()
        # dc/da = 1/a
        assert abs(a.grad - 0.5) < 1e-12

    def test_log_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="log requires positive"):
            log(Tensor(data=0.0, requires_grad=True))

    def test_log_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="log requires positive"):
            log(Tensor(data=-1.0, requires_grad=True))


# ---------------------------------------------------------------------- #
# ReLU
# ---------------------------------------------------------------------- #


class TestRelu:
    def test_relu_positive(self) -> None:
        a = Tensor(data=3.0, requires_grad=True)
        c = relu(a)
        assert c.data == 3.0
        c.backward()
        assert a.grad == 1.0

    def test_relu_negative(self) -> None:
        a = Tensor(data=-2.0, requires_grad=True)
        c = relu(a)
        assert c.data == 0.0
        c.backward()
        assert a.grad == 0.0

    def test_relu_zero(self) -> None:
        a = Tensor(data=0.0, requires_grad=True)
        c = relu(a)
        c.backward()
        assert a.grad == 0.0


# ---------------------------------------------------------------------- #
# Numerical gradient check (the gold standard)
# ---------------------------------------------------------------------- #


class TestNumericalGradientCheck:
    """Compare analytical gradients to numerical ones for composed expressions."""

    @pytest.mark.parametrize(
        "a_val, b_val",
        [(2.0, 3.0), (-1.0, 4.0), (0.5, 0.5), (10.0, -2.0)],
    )
    def test_quadratic_gradients(self, a_val: float, b_val: float) -> None:
        """f = (a + b) * (a - b)  →  ∂f/∂a = 2a,  ∂f/∂b = -2b."""
        a = Tensor(data=a_val, requires_grad=True)
        b = Tensor(data=b_val, requires_grad=True)
        c = (a + b) * (a - b)
        c.backward()
        expected_da = 2.0 * a_val
        expected_db = -2.0 * b_val
        assert abs(a.grad - expected_da) < 1e-9
        assert abs(b.grad - expected_db) < 1e-9

    def test_chain_grad_sin(self) -> None:
        """f = exp(2x)  →  df/dx = 2 * exp(2x)."""
        x_val = 1.5
        x = Tensor(data=x_val, requires_grad=True)
        y = exp(mul(Tensor(data=2.0, requires_grad=False), x))
        y.backward()
        expected = 2.0 * math.exp(2.0 * x_val)
        assert abs(x.grad - expected) < 1e-9

    def test_nested_composition(self) -> None:
        """f = (a * b + c) ** 2  where c is constant.

        Verified by comparing analytical vs numerical.
        """
        random.seed(42)
        a_val, b_val = 1.7, -0.6
        a = Tensor(data=a_val, requires_grad=True)
        b = Tensor(data=b_val, requires_grad=True)
        c = Tensor(data=2.5, requires_grad=False)
        prod = mul(a, b)
        s = add(prod, c)
        out = mul(s, s)  # (ab + c)^2
        out.backward()

        def f_a(x: float) -> float:
            return (x * b_val + 2.5) ** 2

        def f_b(x: float) -> float:
            return (a_val * x + 2.5) ** 2

        assert abs(a.grad - numerical_grad(f_a, a_val)) < 1e-4
        assert abs(b.grad - numerical_grad(f_b, b_val)) < 1e-4


# ---------------------------------------------------------------------- #
# no_grad context
# ---------------------------------------------------------------------- #


class TestNoGrad:
    def test_no_grad_disables_tracking(self) -> None:
        with no_grad():
            a = Tensor(data=2.0, requires_grad=True)
            b = Tensor(data=3.0, requires_grad=True)
            c = a + b
        # The intermediate ``c`` should have requires_grad=False
        # because the context disabled tracking.
        assert c.requires_grad is False

    def test_no_grad_restored_after_block(self) -> None:
        with no_grad():
            pass
        a = Tensor(data=2.0, requires_grad=True)
        b = Tensor(data=3.0, requires_grad=True)
        c = a + b
        # Outside the block, tracking is back on.
        assert c.requires_grad is True

    def test_nested_no_grad(self) -> None:
        with no_grad():
            with no_grad():
                a = Tensor(data=1.0, requires_grad=True)
                b = Tensor(data=1.0, requires_grad=True)
                c = a + b
            assert c.requires_grad is False
        assert c.requires_grad is False


# ---------------------------------------------------------------------- #
# Parameter
# ---------------------------------------------------------------------- #


class TestParameter:
    def test_default_requires_grad(self) -> None:
        p = Parameter(0.5)
        assert p.requires_grad is True

    def test_is_tensor_subclass(self) -> None:
        p = Parameter(0.5)
        assert isinstance(p, Tensor)

    def test_accumulates_grad(self) -> None:
        p = Parameter(2.0)
        loss = mul(p, p)  # p^2
        loss.backward()
        assert p.grad == 4.0  # 2*p = 4


# ---------------------------------------------------------------------- #
# matmul
# ---------------------------------------------------------------------- #


class TestMatmul:
    def test_2x2_times_2x2(self) -> None:
        a = [
            [Tensor(data=1.0, requires_grad=True), Tensor(data=2.0, requires_grad=True)],
            [Tensor(data=3.0, requires_grad=True), Tensor(data=4.0, requires_grad=True)],
        ]
        b = [
            [Tensor(data=5.0, requires_grad=True), Tensor(data=6.0, requires_grad=True)],
            [Tensor(data=7.0, requires_grad=True), Tensor(data=8.0, requires_grad=True)],
        ]
        out = matmul(a, b)
        assert len(out) == 2
        assert all(len(row) == 2 for row in out)
        assert out[0][0].data == 19.0
        assert out[0][1].data == 22.0
        assert out[1][0].data == 43.0
        assert out[1][1].data == 50.0

    def test_empty_inputs(self) -> None:
        assert matmul([], [[Tensor(data=1.0)]]) == []
        assert matmul([[Tensor(data=1.0)]], []) == []

    def test_shape_mismatch_raises(self) -> None:
        a = [[Tensor(data=1.0), Tensor(data=2.0), Tensor(data=3.0)]]
        b = [[Tensor(data=1.0), Tensor(data=2.0)]]
        with pytest.raises(ValueError, match="shape mismatch"):
            matmul(a, b)

    def test_grad_flows_through_matmul(self) -> None:
        """2x2 matmul: gradient on each input should equal the corresponding
        element of the other matrix (the simple 1x1 case scales)."""
        a = [
            [Tensor(data=1.0, requires_grad=True), Tensor(data=2.0, requires_grad=True)],
            [Tensor(data=3.0, requires_grad=True), Tensor(data=4.0, requires_grad=True)],
        ]
        b = [
            [Tensor(data=1.0, requires_grad=True), Tensor(data=0.0, requires_grad=True)],
            [Tensor(data=0.0, requires_grad=True), Tensor(data=1.0, requires_grad=True)],
        ]
        out = matmul(a, b)  # identity -> out == a
        loss = out[0][0] + out[1][1]  # = a[0][0] + a[1][1]
        loss.backward()
        # d loss / d a[0][0] = 1, d loss / d a[1][1] = 1, others 0.
        assert a[0][0].grad == 1.0
        assert a[0][1].grad == 0.0
        assert a[1][0].grad == 0.0
        assert a[1][1].grad == 1.0


# ---------------------------------------------------------------------- #
# Power operator
# ---------------------------------------------------------------------- #


class TestPow:
    def test_squared_grad(self) -> None:
        a = Tensor(data=3.0, requires_grad=True)
        c = a**2
        c.backward()
        # dc/da = 2a
        assert abs(a.grad - 6.0) < 1e-12

    def test_cubed_grad(self) -> None:
        a = Tensor(data=2.0, requires_grad=True)
        c = a**3
        c.backward()
        # dc/da = 3 a^2
        assert abs(a.grad - 12.0) < 1e-12

    def test_non_constant_exponent_raises(self) -> None:
        a = Tensor(data=2.0, requires_grad=True)
        b = Tensor(data=3.0, requires_grad=True)
        with pytest.raises(TypeError, match="constant"):
            a**b  # type: ignore[operator]
