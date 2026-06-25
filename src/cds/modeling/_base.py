"""Abstract base and helpers for the symbolic expression AST.

The symbolic algebra lives in :mod:`cds.modeling._nodes`; this module holds
only the :class:`Expression` abstract base and the two module-level helpers
(``_coerce`` and ``_subs``) that every node type depends on. Splitting them
out keeps the node definitions in one focused file while the base contract —
the operator overloads, ``simplify``/``subs``/``to_func`` convenience layer,
and the abstract ``evaluate``/``diff``/``variables``/``to_str``/``to_latex``
interface — reads as a single self-contained unit here.

The concrete node classes (:class:`Add`, :class:`Constant`, ...) are imported
*lazily inside the operator dunders and helpers* rather than at module top
level. This breaks what would otherwise be a circular import — ``_nodes``
needs :class:`Expression` from this module to define its subclasses, while the
dunders here need the node classes to construct their results. Because the
dunders only run at call time (never at class-definition time), the lazy
import is resolved after both modules are fully loaded, so the cycle never
materialises.
"""

from __future__ import annotations

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
    # Each dunder coerces the plain-number operand to a :class:`Constant` and
    # returns the matching AST node; no mutation, no evaluation. The concrete
    # node classes are imported inside each body (see module docstring for the
    # cycle-breaking rationale).
    # ------------------------------------------------------------------ #
    def __add__(self, other: Expression | float | int) -> Expression:
        """Return ``self + other`` as an :class:`Add` node."""
        from cds.modeling._nodes import Add

        return Add(self, _coerce(other))

    def __radd__(self, other: float | int) -> Expression:
        """Return ``other + self`` (``other`` is a number) as an :class:`Add` node."""
        from cds.modeling._nodes import Add

        return Add(_coerce(other), self)

    def __sub__(self, other: Expression | float | int) -> Expression:
        """Return ``self - other`` as a :class:`Sub` node."""
        from cds.modeling._nodes import Sub

        return Sub(self, _coerce(other))

    def __rsub__(self, other: float | int) -> Expression:
        """Return ``other - self`` (``other`` is a number) as a :class:`Sub` node."""
        from cds.modeling._nodes import Sub

        return Sub(_coerce(other), self)

    def __mul__(self, other: Expression | float | int) -> Expression:
        """Return ``self * other`` as a :class:`Mul` node."""
        from cds.modeling._nodes import Mul

        return Mul(self, _coerce(other))

    def __rmul__(self, other: float | int) -> Expression:
        """Return ``other * self`` (``other`` is a number) as a :class:`Mul` node."""
        from cds.modeling._nodes import Mul

        return Mul(_coerce(other), self)

    def __truediv__(self, other: Expression | float | int) -> Expression:
        """Return ``self / other`` as a :class:`Div` node."""
        from cds.modeling._nodes import Div

        return Div(self, _coerce(other))

    def __rtruediv__(self, other: float | int) -> Expression:
        """Return ``other / self`` (``other`` is a number) as a :class:`Div` node."""
        from cds.modeling._nodes import Div

        return Div(_coerce(other), self)

    def __pow__(self, exponent: Expression | float | int) -> Expression:
        """Return ``self ** exponent`` as a :class:`Pow` node."""
        from cds.modeling._nodes import Pow

        return Pow(self, _coerce(exponent))

    def __rpow__(self, base: float | int) -> Expression:
        """Return ``base ** self`` (``base`` is a number) as a :class:`Pow` node."""
        from cds.modeling._nodes import Pow

        return Pow(_coerce(base), self)

    def __neg__(self) -> Expression:
        """Return the unary negation ``-self`` as ``(-1) * self``."""
        from cds.modeling._nodes import Constant, Mul

        return Mul(Constant(-1.0), self)

    def __pos__(self) -> Expression:
        """Return ``+self`` (a no-op copy of this node)."""
        return self

    def __repr__(self) -> str:
        """Return a developer-facing ``ClassName(<expr>)`` rendering."""
        return f"{self.__class__.__name__}({self.to_str()})"


def _coerce(value: Expression | float | int) -> Expression:
    """Promote a Python number to :class:`Constant`; pass through an Expression."""
    from cds.modeling._nodes import Constant

    if isinstance(value, Expression):
        return value
    if isinstance(value, (int, float)):
        return Constant(float(value))
    raise TypeError(
        f"unsupported operand type {type(value).__name__!r}; expected Expression or number"
    )


def _subs(expr: Expression, values: dict[str, float]) -> Expression:
    """Recursively substitute variables named in ``values`` with Constants."""
    from cds.modeling._nodes import Constant, Variable, _Binary, _Unary

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
