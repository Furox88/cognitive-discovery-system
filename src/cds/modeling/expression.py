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
>>> expr.evaluate({"x": 2.0})
10.0
>>> expr.diff("x").evaluate({"x": 2.0})
7.0

The AST is deliberately decoupled from autograd — either engine can be used
on its own, and they share no code. Pick the symbolic engine when you need
the *formula* of a derivative (analysis, equation development, LaTeX export);
pick autograd when you only need the *number* (training a model).

Implementation is split across two private submodules for readability:
:mod:`cds.modeling._base` holds the :class:`Expression` abstract base and the
``_coerce``/``_subs`` helpers; :mod:`cds.modeling._nodes` holds every concrete
leaf and operator node. This module re-exports the union so the public
``cds.modeling.expression`` API (and the legacy ``from cds.modeling.expression
import ...`` import path) stays stable.

References:
    - Knuth, D.E. (1962). "Two-way equational deduction" — the precedence
      and printing conventions used by :meth:`Expression.to_str`.
    - GHC's ``deriving`` for symbolic algebra — the constant-folding rules
      in :meth:`Expression.simplify` mirror standard textbook identities.
"""

from __future__ import annotations

# Re-export the complete public surface. `_base` and `_nodes` are the real
# home of these symbols; importing them here keeps the long-standing
# ``from cds.modeling.expression import Variable`` path working unchanged.
from cds.modeling._base import Expression
from cds.modeling._nodes import (
    Add,
    Constant,
    Cos,
    Div,
    Exp,
    Log,
    Mul,
    Pow,
    Sin,
    Sqrt,
    Sub,
    Variable,
)

__all__ = [
    "Expression",
    "Constant",
    "Variable",
    "Add",
    "Sub",
    "Mul",
    "Div",
    "Pow",
    "Sin",
    "Cos",
    "Exp",
    "Log",
    "Sqrt",
]
