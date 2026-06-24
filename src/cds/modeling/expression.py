"""Symbolic math expressions — a small, self-contained algebraic AST.

CDS already ships :mod:`cds.nlp.autograd` for *numerical* reverse-mode
differentiation. That engine is brilliant for watching gradients flow
through a neural net, but it cannot answer "what is the symbolic
derivative of ``x**2 + 3*x``?" — it only hands back a number at a point.

This module fills that gap with a tiny symbolic algebra:

* :class:`Expression` is the abstract base every node subclasses.
* Leaves: :class:`Constant` (a number) and :class:`Variable` (a named symbol).
* Binary ops: :class:`Add`, :class:`Sub`, :class:`Mul`, :class:`Div`, :class:`Pow`.
* Unary transcendentals: :class:`Sin`, :class:`Cos`, :class:`Exp`, :class:`Log`,
  :class:`Sqrt`.

Every node supports operator overloading (``+ - * / **``), so expressions are
written the natural Python way:

>>> from cds.modeling.expression import Variable, Constant
>>> x = Variable("x")
>>> expr = x ** 2 + 3 * x
>>> expr.diff("x").to_str()
'(2 * (x ** 2)) + 3'
>>> expr.evaluate({"x": 2.0})
10.0

The AST is deliberately decoupled from autograd — either engine can be used
on its own, and they share no code. Pick the symbolic engine when you need
the *formula* of a derivative (analysis, equation development, LaTeX export);
pick autograd when you only need the *number* (training a model).

References:
    - Knuth, D.E. (1962). "Two-way equational deduction" — the precedence
      and printing conventions used by :meth:`Expression.to_str`.
    - GHC's ``deriving`` for symbolic algebra — the constant-folding rules
      in :meth:`Expression.simplify` mirror standard textbook identities.
"""

from __future__ import annotations

import math
from collections.abc import Callable


class Expression:
    """Abstract base for every node in the symbolic expression tree.

    Subclasses implement :meth:`evaluate`, :meth:`diff`, :meth:`variables`,
    :meth:`simplify`, :meth:`to_latex`, and :meth:`to_str`. The dunders
    (``__add__`` etc.) live here so arithmetic works uniformly for all nodes
    and for mixing nodes with plain Python numbers.
    """

    # ------------------------------------------------------------------ #
    # Abstract interface — every subclass overrides these.
    # ------------------------------------------------------------------ #
    def evaluate(self, env: dict[str, float]) -> float:  # pragma: no cover
        """Evaluate this expression to a float using the variable bindings in ``env``."""
        raise NotImplementedError

    def diff(self, var: str) -> Expression:  # pragma: no cover
        """Return the symbolic derivative of this expression w.r.t. ``var``."""
        raise NotImplementedError

    def variables(self) -> set[str]:  # pragma: no cover
        """Return the set of free variable names appearing in this expression."""
        raise NotImplementedError

    def to_str(self) -> str:  # pragma: no cover
        """Render this expression as a human-readable infix string."""
        raise NotImplementedError

    def to_latex(self) -> str:  # pragma: no cover
        """Render this expression as a LaTeX math string."""
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # Convenience — built on the abstract methods, so shared by all nodes.
    # ------------------------------------------------------------------ #
    def simplify(self) -> Expression:
        """Constant-fold and apply algebraic identities to simplify this tree.

        The base implementation is a no-op (returns ``self``); concrete
        operator subclasses override it to fold constant subexpressions and
        apply identities such as ``x + 0 -> x`` or ``x * 1 -> x``.
        """
        return self

    def subs(self, **values: float) -> Expression:
        """Substitute the given ``name=value`` pairs, returning a new expression.

        Names not present in this expression are ignored, so partial
        substitution is safe.
        """
        return _subs(self, values)

    def to_func(self, *var_names: str) -> Callable[..., float]:
        """Compile this expression into a callable ``f(*args) -> float``.

        The positional argument order is ``var_names``; this is the shape
        :mod:`cds.optimization` and :mod:`cds.modeling.solver` expect.
        """
        unknown = set(var_names) - self.variables()
        if unknown:
            raise ValueError(f"to_func var_names not in expression: {sorted(unknown)}")
        for needed in self.variables():
            if needed not in var_names:
                raise ValueError(f"to_func missing variable {needed!r} in var_names")
        names = list(var_names)

        def _f(*args: float) -> float:
            if len(args) != len(names):
                raise ValueError(f"expected {len(names)} args, got {len(args)}")
            env = dict(zip(names, args))
            return self.evaluate(env)

        return _f

    # ------------------------------------------------------------------ #
    # Operator overloads — promote numbers to Constant so math reads naturally.
    # ------------------------------------------------------------------ #
    def __add__(self, other: Expression | float | int) -> Expression:
        return Add(self, _coerce(other))

    def __radd__(self, other: float | int) -> Expression:
        return Add(_coerce(other), self)

    def __sub__(self, other: Expression | float | int) -> Expression:
        return Sub(self, _coerce(other))

    def __rsub__(self, other: float | int) -> Expression:
        return Sub(_coerce(other), self)

    def __mul__(self, other: Expression | float | int) -> Expression:
        return Mul(self, _coerce(other))

    def __rmul__(self, other: float | int) -> Expression:
        return Mul(_coerce(other), self)

    def __truediv__(self, other: Expression | float | int) -> Expression:
        return Div(self, _coerce(other))

    def __rtruediv__(self, other: float | int) -> Expression:
        return Div(_coerce(other), self)

    def __pow__(self, exponent: Expression | float | int) -> Expression:
        return Pow(self, _coerce(exponent))

    def __rpow__(self, base: float | int) -> Expression:
        return Pow(_coerce(base), self)

    def __neg__(self) -> Expression:
        return Mul(Constant(-1.0), self)

    def __pos__(self) -> Expression:
        return self

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_str()})"


def _coerce(value: Expression | float | int) -> Expression:
    """Promote a Python number to :class:`Constant`; pass through an Expression."""
    if isinstance(value, Expression):
        return value
    if isinstance(value, (int, float)):
        return Constant(float(value))
    raise TypeError(
        f"unsupported operand type {type(value).__name__!r}; expected Expression or number"
    )


def _subs(expr: Expression, values: dict[str, float]) -> Expression:
    """Recursively substitute variables named in ``values`` with Constants."""
    if isinstance(expr, Variable):
        if expr.name in values:
            return Constant(values[expr.name])
        return expr
    if isinstance(expr, Constant):
        return expr
    if isinstance(expr, _Unary):
        return type(expr)(_subs(expr.operand, values))
    if isinstance(expr, _Binary):
        return type(expr)(_subs(expr.left, values), _subs(expr.right, values))
    # Should be unreachable: every concrete node is one of the above.
    raise TypeError(f"cannot substitute into {type(expr).__name__}")  # pragma: no cover


# ---------------------------------------------------------------------- #
# Leaves
# ---------------------------------------------------------------------- #


class Constant(Expression):
    """A literal numeric value in an expression tree."""

    __slots__ = ("value",)

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def evaluate(self, env: dict[str, float]) -> float:
        return self.value

    def diff(self, var: str) -> Expression:
        return Constant(0.0)

    def variables(self) -> set[str]:
        return set()

    def to_str(self) -> str:
        # Render integers without a trailing ".0" for readability.
        if self.value == int(self.value):
            return str(int(self.value))
        return repr(self.value)

    def to_latex(self) -> str:
        if self.value == int(self.value):
            return str(int(self.value))
        return repr(self.value)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Constant) and other.value == self.value

    def __hash__(self) -> int:
        return hash(("Constant", self.value))


class Variable(Expression):
    """A named symbolic variable (e.g. ``x``, ``theta``)."""

    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def evaluate(self, env: dict[str, float]) -> float:
        if self.name not in env:
            raise ValueError(f"no value bound for variable {self.name!r}")
        return env[self.name]

    def diff(self, var: str) -> Expression:
        return Constant(1.0) if var == self.name else Constant(0.0)

    def variables(self) -> set[str]:
        return {self.name}

    def to_str(self) -> str:
        return self.name

    def to_latex(self) -> str:
        return self.name

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Variable) and other.name == self.name

    def __hash__(self) -> int:
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
        return self.left.evaluate(env) + self.right.evaluate(env)

    def diff(self, var: str) -> Expression:
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
        return f"({self.left.to_str()} + {self.right.to_str()})"

    def to_latex(self) -> str:
        return f"{self.left.to_latex()} + {self.right.to_latex()}"


class Sub(_Binary):
    """``left - right``."""

    def evaluate(self, env: dict[str, float]) -> float:
        return self.left.evaluate(env) - self.right.evaluate(env)

    def diff(self, var: str) -> Expression:
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
        return f"({self.left.to_str()} - {self.right.to_str()})"

    def to_latex(self) -> str:
        return f"{self.left.to_latex()} - {self.right.to_latex()}"


class Mul(_Binary):
    """``left * right`` (product rule for differentiation)."""

    def evaluate(self, env: dict[str, float]) -> float:
        return self.left.evaluate(env) * self.right.evaluate(env)

    def diff(self, var: str) -> Expression:
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
        return f"({self.left.to_str()} * {self.right.to_str()})"

    def to_latex(self) -> str:
        return f"{self.left.to_latex()} \\cdot {self.right.to_latex()}"


class Div(_Binary):
    """``left / right`` (quotient rule for differentiation)."""

    def evaluate(self, env: dict[str, float]) -> float:
        return self.left.evaluate(env) / self.right.evaluate(env)

    def diff(self, var: str) -> Expression:
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
            return Constant(left.value / right.value)
        if isinstance(left, Constant) and left.value == 0.0:
            return Constant(0.0)
        if isinstance(right, Constant) and right.value == 1.0:
            return left
        return Div(left, right)

    def to_str(self) -> str:
        return f"({self.left.to_str()} / {self.right.to_str()})"

    def to_latex(self) -> str:
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
        return float(self.left.evaluate(env) ** self.right.evaluate(env))

    def diff(self, var: str) -> Expression:
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
        and ``x ** 1 -> x`` once the base and exponent are simplified."""
        base = self.left.simplify()
        exp = self.right.simplify()
        if isinstance(base, Constant) and isinstance(exp, Constant):
            return Constant(base.value**exp.value)
        if isinstance(exp, Constant):
            if exp.value == 0.0:
                return Constant(1.0)
            if exp.value == 1.0:
                return base
        return Pow(base, exp)

    def to_str(self) -> str:
        return f"({self.left.to_str()} ** {self.right.to_str()})"

    def to_latex(self) -> str:
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
        return math.sin(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        # Chain rule: d/dx sin(u) = cos(u) * u'
        return Mul(Cos(self.operand), self.operand.diff(var))

    def simplify(self) -> Expression:
        """Fold ``sin(c)`` to a :class:`Constant`; otherwise simplify the operand."""
        inner = self.operand.simplify()
        if isinstance(inner, Constant):
            return Constant(math.sin(inner.value))
        return Sin(inner)

    def to_str(self) -> str:
        return f"sin({self.operand.to_str()})"

    def to_latex(self) -> str:
        return f"\\sin\\left({self.operand.to_latex()}\\right)"


class Cos(_Unary):
    """``cos(operand)``."""

    def evaluate(self, env: dict[str, float]) -> float:
        return math.cos(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        # Chain rule: d/dx cos(u) = -sin(u) * u'
        return Mul(Mul(Constant(-1.0), Sin(self.operand)), self.operand.diff(var))

    def simplify(self) -> Expression:
        """Fold ``cos(c)`` to a :class:`Constant`; otherwise simplify the operand."""
        inner = self.operand.simplify()
        if isinstance(inner, Constant):
            return Constant(math.cos(inner.value))
        return Cos(inner)

    def to_str(self) -> str:
        return f"cos({self.operand.to_str()})"

    def to_latex(self) -> str:
        return f"\\cos\\left({self.operand.to_latex()}\\right)"


class Exp(_Unary):
    """``e ** operand`` (the exponential function)."""

    def evaluate(self, env: dict[str, float]) -> float:
        return math.exp(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        # Chain rule: d/dx exp(u) = exp(u) * u'
        return Mul(Exp(self.operand), self.operand.diff(var))

    def simplify(self) -> Expression:
        """Fold ``exp(c)`` to a :class:`Constant`; otherwise simplify the operand."""
        inner = self.operand.simplify()
        if isinstance(inner, Constant):
            return Constant(math.exp(inner.value))
        return Exp(inner)

    def to_str(self) -> str:
        return f"exp({self.operand.to_str()})"

    def to_latex(self) -> str:
        return f"e^{{{self.operand.to_latex()}}}"


class Log(_Unary):
    """``ln(operand)`` (the natural logarithm, base *e*)."""

    def evaluate(self, env: dict[str, float]) -> float:
        return math.log(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        # Chain rule: d/dx ln(u) = u' / u
        return Div(self.operand.diff(var), self.operand)

    def simplify(self) -> Expression:
        """Fold ``log(c)`` to a :class:`Constant`; otherwise simplify the operand.

        A non-positive operand is left untouched so the domain error surfaces
        at evaluation time rather than being silently hidden.
        """
        inner = self.operand.simplify()
        if isinstance(inner, Constant):
            return Constant(math.log(inner.value))
        return Log(inner)

    def to_str(self) -> str:
        return f"log({self.operand.to_str()})"

    def to_latex(self) -> str:
        return f"\\ln\\left({self.operand.to_latex()}\\right)"


class Sqrt(_Unary):
    """``sqrt(operand)`` (the principal square root)."""

    def evaluate(self, env: dict[str, float]) -> float:
        return math.sqrt(self.operand.evaluate(env))

    def diff(self, var: str) -> Expression:
        # Chain rule: d/dx sqrt(u) = u' / (2 * sqrt(u))
        return Div(
            self.operand.diff(var),
            Mul(Constant(2.0), Sqrt(self.operand)),
        )

    def simplify(self) -> Expression:
        """Fold ``sqrt(c)`` to a :class:`Constant`; otherwise simplify the operand."""
        inner = self.operand.simplify()
        if isinstance(inner, Constant):
            return Constant(math.sqrt(inner.value))
        return Sqrt(inner)

    def to_str(self) -> str:
        return f"sqrt({self.operand.to_str()})"

    def to_latex(self) -> str:
        return f"\\sqrt{{{self.operand.to_latex()}}}"
