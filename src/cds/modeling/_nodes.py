"""Concrete node classes for the symbolic expression AST.

This module defines the leaves (:class:`Constant`, :class:`Variable`), the
shared binary/unary scaffolding (:class:`_Binary`, :class:`_Unary`), and every
operator node — the binary ops ``Add``/``Sub``/``Mul``/``Div``/``Pow`` and the
unary transcendentals ``Sin``/``Cos``/``Exp``/``Log``/``Sqrt``. Each node
implements the contract declared by :class:`cds.modeling._base.Expression`:
``evaluate``, ``diff``, ``variables``, ``simplify``, ``to_str``, and
``to_latex``.

The base class and its operator overloads live in :mod:`cds.modeling._base`;
only :class:`Expression` is imported from there, so this module has a single,
acyclic dependency on the base.
"""

from __future__ import annotations

import math

from cds.modeling._base import Expression

# ---------------------------------------------------------------------- #
# Leaves
# ---------------------------------------------------------------------- #


class Constant(Expression):
    """A literal numeric value in an expression tree."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        """Store the literal value, coerced to ``float``."""
        self.value = float(value)

    def evaluate(self, env: dict[str, float]) -> float:
        """Return the constant's value (independent of ``env``)."""
        return self.value

    def diff(self, var: str) -> Expression:
        """The derivative of a constant is zero."""
        return Constant(0.0)

    def variables(self) -> set[str]:
        """A constant binds no variables."""
        return set()

    def to_str(self) -> str:
        """Render the value, dropping the ``.0`` for whole numbers."""
        # Render integers without a trailing ".0" for readability.
        if self.value == int(self.value):
            return str(int(self.value))
        return repr(self.value)

    def to_latex(self) -> str:
        """Render the value in LaTeX, dropping the ``.0`` for whole numbers."""
        if self.value == int(self.value):
            return str(int(self.value))
        return repr(self.value)

    def __eq__(self, other: object) -> bool:
        """Two constants are equal iff their values are equal."""
        return isinstance(other, Constant) and other.value == self.value

    def __hash__(self) -> int:
        """Hash keyed on the ``Constant`` tag and value (matches ``__eq__``)."""
        return hash(("Constant", self.value))


class Variable(Expression):
    """A named symbolic variable (e.g. ``x``, ``theta``)."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        """Store the variable's symbolic ``name``."""
        self.name = name

    def evaluate(self, env: dict[str, float]) -> float:
        """Look up this variable's value in ``env`` (raises if unbound)."""
        if self.name not in env:
            raise ValueError(f"no value bound for variable {self.name!r}")
        return env[self.name]

    def diff(self, var: str) -> Expression:
        """Derivative is ``1`` w.r.t. itself, ``0`` otherwise."""
        return Constant(1.0) if var == self.name else Constant(0.0)

    def variables(self) -> set[str]:
        """This variable's own name."""
        return {self.name}

    def to_str(self) -> str:
        """Render the variable name verbatim."""
        return self.name

    def to_latex(self) -> str:
        """Render the variable name verbatim."""
        return self.name

    def __eq__(self, other: object) -> bool:
        """Two variables are equal iff their names match."""
        return isinstance(other, Variable) and other.name == self.name

    def __hash__(self) -> int:
        """Hash keyed on the ``Variable`` tag and name (matches ``__eq__``)."""
        return hash(("Variable", self.name))


# ---------------------------------------------------------------------- #
# Binary ops
# ---------------------------------------------------------------------- #


class _Binary(Expression):
    """Shared scaffolding for two-argument operators."""

    __slots__ = ("left", "right")

    def __init__(self, left: Expression, right: Expression) -> None:
        self.left = left
        self.right = right

    def variables(self) -> set[str]:
        return self.left.variables() | self.right.variables()


class Add(_Binary):
    """``left + right``."""

    def evaluate(self, env: dict[str, float]) -> float:
        """Return the numeric sum of both operands under ``env``."""
        return self.left.evaluate(env) + self.right.evaluate(env)

    def diff(self, var: str) -> Expression:
        """Sum rule: ``d/dx (u + v) = u' + v'``."""
        return Add(self.left.diff(var), self.right.diff(var))

    def simplify(self) -> Expression:
        """Fold ``c + c`` and drop additive zeros (``0 + x`` or ``x + 0``)."""
        left = self.left.simplify()
        right = self.right.simplify()
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value + right.value)
        if isinstance(left, Constant) and left.value == 0.0:
            return right
        if isinstance(right, Constant) and right.value == 0.0:
            return left
        return Add(left, right)

    def to_str(self) -> str:
        """Render the sum as ``(left + right)``."""
        return f"({self.left.to_str()} + {self.right.to_str()})"

    def to_latex(self) -> str:
        """Render the sum in LaTeX as ``left + right``."""
        return f"{self.left.to_latex()} + {self.right.to_latex()}"


class Sub(_Binary):
    """``left - right``."""

    def evaluate(self, env: dict[str, float]) -> float:
        """Return the numeric difference of both operands under ``env``."""
        return self.left.evaluate(env) - self.right.evaluate(env)

    def diff(self, var: str) -> Expression:
        """Difference rule: ``d/dx (u - v) = u' - v'``."""
        return Sub(self.left.diff(var), self.right.diff(var))

    def simplify(self) -> Expression:
        """Fold ``c - c`` and drop a subtractive zero (``x - 0``).

        Note that ``0 - x`` is *not* rewritten to ``-x`` here — that would
        need a sign negation not expressible as a plain :class:`Sub`.
        """
        left = self.left.simplify()
        right = self.right.simplify()
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value - right.value)
        if isinstance(right, Constant) and right.value == 0.0:
            return left
        return Sub(left, right)

    def to_str(self) -> str:
        """Render the difference as ``(left - right)``."""
        return f"({self.left.to_str()} - {self.right.to_str()})"

    def to_latex(self) -> str:
        """Render the difference in LaTeX as ``left - right``."""
        return f"{self.left.to_latex()} - {self.right.to_latex()}"


class Mul(_Binary):
    """``left * right`` (product rule for differentiation)."""

    def evaluate(self, env: dict[str, float]) -> float:
        """Return the numeric product of both operands under ``env``."""
        return self.left.evaluate(env) * self.right.evaluate(env)

    def diff(self, var: str) -> Expression:
        """Product rule: ``d/dx (u * v) = u'*v + u*v'``."""
        # Product rule: d(uv) = u'v + uv'
        return Add(
            Mul(self.left.diff(var), self.right),
            Mul(self.left, self.right.diff(var)),
        )

    def simplify(self) -> Expression:
        """Fold ``c * c``, collapse a zero factor (``0 * x`` -> ``0``), and drop
        multiplicative identities (``1 * x`` or ``x * 1`` -> ``x``)."""
        left = self.left.simplify()
        right = self.right.simplify()
        if isinstance(left, Constant) and isinstance(right, Constant):
            return Constant(left.value * right.value)
        if isinstance(left, Constant):
            if left.value == 0.0:
                return Constant(0.0)
            if left.value == 1.0:
                return right
        if isinstance(right, Constant):
            if right.value == 0.0:
                return Constant(0.0)
            if right.value == 1.0:
                return left
        return Mul(left, right)

    def to_str(self) -> str:
        """Render the product as ``(left * right)``."""
        return f"({self.left.to_str()} * {self.right.to_str()})"

    def to_latex(self) -> str:
        """Render the product in LaTeX as ``left \\cdot right``."""
        return f"{self.left.to_latex()} \\cdot {self.right.to_latex()}"


class Div(_Binary):
    """``left / right`` (quotient rule for differentiation)."""

    def evaluate(self, env: dict[str, float]) -> float:
        """Return the numeric quotient of both operands under ``env``."""
        return self.left.evaluate(env) / self.right.evaluate(env)

    def diff(self, var: str) -> Expression:
        """Quotient rule: ``d/dx (u / v) = (u'*v - u*v') / v^2``."""
        # Quotient rule: d(u/v) = (u'v - uv') / v^2
        return Div(
            Sub(
                Mul(self.left.diff(var), self.right),
                Mul(self.left, self.right.diff(var)),
            ),
            Pow(self.right, Constant(2.0)),
        )

    def simplify(self) -> Expression:
        """Fold ``c / c``, collapse a zero numerator (``0 / x`` -> ``0``), and
        drop a unit denominator (``x / 1`` -> ``x``). A zero divisor is left
        in place to surface a real error at evaluation time."""
        left = self.left.simplify()
        right = self.right.simplify()
        if isinstance(left, Constant) and isinstance(right, Constant):
            # A zero divisor is undefined: leave it as a Div so the
            # ZeroDivisionError surfaces at evaluation time, not here.
            if right.value == 0.0:
                return Div(left, right)
            return Constant(left.value / right.value)
        if isinstance(left, Constant) and left.value == 0.0:
            return Constant(0.0)
        if isinstance(right, Constant) and right.value == 1.0:
            return left
        return Div(left, right)

    def to_str(self) -> str:
        """Render the quotient as ``(left / right)``."""
        return f"({self.left.to_str()} / {self.right.to_str()})"

    def to_latex(self) -> str:
        """Render the quotient in LaTeX as ``\\frac{left}{right}``."""
        return f"\\frac{{{self.left.to_latex()}}}{{{self.right.to_latex()}}}"


class Pow(_Binary):
    """``base ** exponent``.

    Differentiation handles two useful cases: a constant exponent
    (``d/dx u^c = c * u^(c-1) * u'``) and a constant base
    (``d/dx c^u = c^u * ln(c) * u'``). The fully general case
    ``u^v`` is handled via logarithmic differentiation:
    ``u^v * (v' * ln(u) + v * u'/u)``.
    """

    def evaluate(self, env: dict[str, float]) -> float:
        """Return the numeric ``left ** right`` under ``env`` (as ``float``)."""
        return float(self.left.evaluate(env) ** self.right.evaluate(env))

    def diff(self, var: str) -> Expression:
        """Differentiate ``u ** v``, dispatching on which side depends on ``var``.

        Covers constant-exponent (power rule), constant-base (exponential
        rule), and the general logarithmic-differentiation case ``u ** v``.
        """
        base = self.left
        exp = self.right
        base_has = var in base.variables()
        exp_has = var in exp.variables()

        if not base_has and not exp_has:
            return Constant(0.0)
        if exp_has and not base_has:
            # d/dx c^u = c^u * ln(c) * u'
            return Mul(
                Mul(Pow(base, exp), Log(base)),
                exp.diff(var),
            )
        if base_has and not exp_has:
            # d/dx u^c = c * u^(c-1) * u'
            return Mul(
                Mul(exp, Pow(base, Sub(exp, Constant(1.0)))),
                base.diff(var),
            )
        # General case u^v: u^v * (v' * ln(u) + v * u'/u)
        return Mul(
            Pow(base, exp),
            Add(
                Mul(exp.diff(var), Log(base)),
                Div(Mul(exp, base.diff(var)), base),
            ),
        )

    def simplify(self) -> Expression:
        """Fold ``c ** c``, and apply the exponent identities ``x ** 0 -> 1``
        and ``x ** 1 -> x`` once the base and exponent are simplified.

        A constant fold that would yield a complex result (a negative base to a
        fractional exponent) is left in place so the error surfaces at
        evaluation time rather than as a ``TypeError`` here."""
        base = self.left.simplify()
        exp = self.right.simplify()
        if isinstance(base, Constant) and isinstance(exp, Constant):
            result = base.value**exp.value
            # A negative base raised to a fractional exponent is complex; leave
            # it as a Pow so the (real) evaluation surfaces the error.
            if isinstance(result, complex):
                return Pow(base, exp)
            return Constant(result)
        if isinstance(exp, Constant):
            if exp.value == 0.0:
                return Constant(1.0)
            if exp.value == 1.0:
                return base
        return Pow(base, exp)

    def to_str(self) -> str:
        """Render the power as ``(left ** right)``."""
        return f"({self.left.to_str()} ** {self.right.to_str()})"

    def to_latex(self) -> str:
        """Render the power in LaTeX as ``left^{right}``."""
        return f"{self.left.to_latex()}^{{{self.right.to_latex()}}}"


# ---------------------------------------------------------------------- #
# Unary transcendentals
# ---------------------------------------------------------------------- #


class _Unary(Expression):
    """Shared scaffolding for single-argument operators."""

    __slots__ = ("operand",)

    def __init__(self, operand: Expression) -> None:
        self.operand = operand

    def variables(self) -> set[str]:
        return self.operand.variables()


class Sin(_Unary):
    """``sin(operand)``."""

    def evaluate(self, env: dict[str, float]) -> float:
        """Return ``math.sin(operand)`` evaluated under ``env``."""
        return math.sin(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        """Chain rule: ``d/dx sin(u) = cos(u) * u'``."""
        # Chain rule: d/dx sin(u) = cos(u) * u'
        return Mul(Cos(self.operand), self.operand.diff(var))

    def simplify(self) -> Expression:
        """Fold ``sin(c)`` to a :class:`Constant`; otherwise simplify the operand."""
        inner = self.operand.simplify()
        if isinstance(inner, Constant):
            return Constant(math.sin(inner.value))
        return Sin(inner)

    def to_str(self) -> str:
        """Render as ``sin(operand)``."""
        return f"sin({self.operand.to_str()})"

    def to_latex(self) -> str:
        """Render in LaTeX as ``\\sin\\left(operand\\right)``."""
        return f"\\sin\\left({self.operand.to_latex()}\\right)"


class Cos(_Unary):
    """``cos(operand)``."""

    def evaluate(self, env: dict[str, float]) -> float:
        """Return ``math.cos(operand)`` evaluated under ``env``."""
        return math.cos(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        """Chain rule: ``d/dx cos(u) = -sin(u) * u'``."""
        # Chain rule: d/dx cos(u) = -sin(u) * u'
        return Mul(Mul(Constant(-1.0), Sin(self.operand)), self.operand.diff(var))

    def simplify(self) -> Expression:
        """Fold ``cos(c)`` to a :class:`Constant`; otherwise simplify the operand."""
        inner = self.operand.simplify()
        if isinstance(inner, Constant):
            return Constant(math.cos(inner.value))
        return Cos(inner)

    def to_str(self) -> str:
        """Render as ``cos(operand)``."""
        return f"cos({self.operand.to_str()})"

    def to_latex(self) -> str:
        """Render in LaTeX as ``\\cos\\left(operand\\right)``."""
        return f"\\cos\\left({self.operand.to_latex()}\\right)"


class Exp(_Unary):
    """``e ** operand`` (the exponential function)."""

    def evaluate(self, env: dict[str, float]) -> float:
        """Return ``math.exp(operand)`` evaluated under ``env``."""
        return math.exp(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        """Chain rule: ``d/dx exp(u) = exp(u) * u'``."""
        # Chain rule: d/dx exp(u) = exp(u) * u'
        return Mul(Exp(self.operand), self.operand.diff(var))

    def simplify(self) -> Expression:
        """Fold ``exp(c)`` to a :class:`Constant`; otherwise simplify the operand."""
        inner = self.operand.simplify()
        if isinstance(inner, Constant):
            return Constant(math.exp(inner.value))
        return Exp(inner)

    def to_str(self) -> str:
        """Render as ``exp(operand)``."""
        return f"exp({self.operand.to_str()})"

    def to_latex(self) -> str:
        """Render in LaTeX as ``e^{operand}``."""
        return f"e^{{{self.operand.to_latex()}}}"


class Log(_Unary):
    """``ln(operand)`` (the natural logarithm, base *e*)."""

    def evaluate(self, env: dict[str, float]) -> float:
        """Return ``math.log(operand)`` evaluated under ``env``."""
        return math.log(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        """Chain rule: ``d/dx ln(u) = u' / u``."""
        # Chain rule: d/dx ln(u) = u' / u
        return Div(self.operand.diff(var), self.operand)

    def simplify(self) -> Expression:
        """Fold ``log(c)`` to a :class:`Constant`; otherwise simplify the operand.

        A non-positive operand is left untouched so the domain error surfaces
        at evaluation time rather than being silently hidden.
        """
        inner = self.operand.simplify()
        if isinstance(inner, Constant) and inner.value > 0.0:
            return Constant(math.log(inner.value))
        return Log(inner)

    def to_str(self) -> str:
        """Render as ``log(operand)``."""
        return f"log({self.operand.to_str()})"

    def to_latex(self) -> str:
        """Render in LaTeX as ``\\ln\\left(operand\\right)``."""
        return f"\\ln\\left({self.operand.to_latex()}\\right)"


class Sqrt(_Unary):
    """``sqrt(operand)`` (the principal square root)."""

    def evaluate(self, env: dict[str, float]) -> float:
        """Return ``math.sqrt(operand)`` evaluated under ``env``."""
        return math.sqrt(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        """Chain rule: ``d/dx sqrt(u) = u' / (2 * sqrt(u))``."""
        # Chain rule: d/dx sqrt(u) = u' / (2 * sqrt(u))
        return Div(
            self.operand.diff(var),
            Mul(Constant(2.0), Sqrt(self.operand)),
        )

    def simplify(self) -> Expression:
        """Fold ``sqrt(c)`` to a :class:`Constant`; otherwise simplify the operand.

        A negative operand is left untouched so the domain error surfaces at
        evaluation time rather than being silently hidden."""
        inner = self.operand.simplify()
        if isinstance(inner, Constant) and inner.value >= 0.0:
            return Constant(math.sqrt(inner.value))
        return Sqrt(inner)

    def to_str(self) -> str:
        """Render as ``sqrt(operand)``."""
        return f"sqrt({self.operand.to_str()})"

    def to_latex(self) -> str:
        """Render in LaTeX as ``\\sqrt{operand}``."""
        return f"\\sqrt{{{self.operand.to_latex()}}}"
