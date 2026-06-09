"""Mathematical modeling utilities.

Provides lightweight tools for defining symbolic expressions, evaluating
them numerically, and performing basic calculus operations (differentiation
and numerical integration) — useful for physics-oriented reasoning
workflows.
"""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass, field
from typing import Callable, Mapping


# Supported binary operators
_OPS: dict[str, Callable[[float, float], float]] = {
    "+": operator.add,
    "-": operator.sub,
    "*": operator.mul,
    "/": operator.truediv,
    "^": operator.pow,
}

# Supported unary functions
_FUNCS: dict[str, Callable[[float], float]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "abs": abs,
}


@dataclass(frozen=True)
class Variable:
    """A named symbolic variable."""

    name: str

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("Variable name must not be empty")

    def evaluate(self, env: Mapping[str, float]) -> float:
        if self.name not in env:
            raise KeyError(f"Variable {self.name!r} not found in environment")
        return env[self.name]


@dataclass(frozen=True)
class Constant:
    """A numeric constant."""

    value: float

    def evaluate(self, env: Mapping[str, float]) -> float:
        return self.value


@dataclass(frozen=True)
class BinaryOp:
    """A binary operation on two sub-expressions."""

    op: str
    left: "Expression"
    right: "Expression"

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"Unsupported operator {self.op!r}")

    def evaluate(self, env: Mapping[str, float]) -> float:
        lv = self.left.evaluate(env)
        rv = self.right.evaluate(env)
        return _OPS[self.op](lv, rv)


@dataclass(frozen=True)
class UnaryFunc:
    """A unary function applied to a sub-expression."""

    func: str
    arg: "Expression"

    def __post_init__(self) -> None:
        if self.func not in _FUNCS:
            raise ValueError(f"Unsupported function {self.func!r}")

    def evaluate(self, env: Mapping[str, float]) -> float:
        return _FUNCS[self.func](self.arg.evaluate(env))


# Union of all expression types
Expression = Variable | Constant | BinaryOp | UnaryFunc


def differentiate_numerically(
    expr: Expression,
    var: str,
    env: Mapping[str, float],
    h: float = 1e-8,
) -> float:
    """Approximate the partial derivative of *expr* w.r.t. *var* via central differences.

    Args:
        expr: The expression to differentiate.
        var: Name of the variable to differentiate with respect to.
        env: Variable-name → value mapping for the evaluation point.
        h: Step size for the finite-difference approximation.

    Returns:
        Approximate value of ∂expr/∂var evaluated at *env*.
    """
    if var not in env:
        raise KeyError(f"Variable {var!r} not found in environment")
    if h <= 0:
        raise ValueError("Step size h must be positive")

    env_plus = {**env, var: env[var] + h}
    env_minus = {**env, var: env[var] - h}
    return (expr.evaluate(env_plus) - expr.evaluate(env_minus)) / (2 * h)


def integrate_numerically(
    expr: Expression,
    var: str,
    a: float,
    b: float,
    env: Mapping[str, float],
    n: int = 1000,
) -> float:
    """Approximate the definite integral of *expr* over [a, b] using the trapezoidal rule.

    Args:
        expr: The expression to integrate.
        var: Name of the integration variable.
        a: Lower bound.
        b: Upper bound.
        env: Additional variable values (overridden by the integration variable).
        n: Number of trapezoids.

    Returns:
        Approximate value of ∫ₐᵇ expr d(var).
    """
    if n <= 0:
        raise ValueError("n must be a positive integer")
    if a > b:
        raise ValueError("Lower bound a must be <= upper bound b")

    h = (b - a) / n
    total = 0.0
    for i in range(n + 1):
        x = a + i * h
        val = expr.evaluate({**env, var: x})
        weight = 1.0 if (i == 0 or i == n) else 2.0
        total += weight * val
    return total * h / 2.0
