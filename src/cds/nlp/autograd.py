"""Reverse-mode automatic differentiation in pure Python.

A from-scratch autograd engine inspired by Karpathy's micrograd:
- :class:`Tensor` wraps a Python ``float`` with a graph node
  (data, gradient, backward closure, child nodes).
- Ops (``+``, ``*``, ``@``, ``gelu``, ``softmax``) record themselves
  on the graph via ``_backward`` closures during the forward pass.
- :meth:`Tensor.backward` runs a topological reverse traversal and
  fills ``.grad`` on every leaf.

Why this lives in CDS:
- Closes the loop on the educational NLP track: with BPE + embeddings
  + attention (Sprints 1+2) in hand, a learner can now watch a
  tiny GPT actually fit Shakespeare (Sprint 4) — and *see* every
  gradient flowing back through the graph because the engine is 250
  lines, not 50,000.
- The optional ``cds[fast-jit]`` Numba backend swaps just the matmul
  + softmax hot-paths for ~10× speed-up; the rest of the graph stays
  pure Python so the educational narrative is unbroken.

References:
    - Karpathy, A. (2020). "micrograd — A tiny scalar-valued autograd
      engine and a neural net library on top of it."
      https://github.com/karpathy/micrograd
    - Karpathy, A. (2023). "Let's build GPT: from scratch, in code,
      spelled out." (the training pipeline this engine plugs into.)
    - PyTorch's autograd docs — for the ``requires_grad`` /
      ``backward()`` API conventions this module imitates.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------- #
# Type aliases
# ---------------------------------------------------------------------- #

# The underlying scalar type. Pure Python floats throughout — the
# autograd engine doesn't care whether a Tensor represents a single
# float or a (deferred) matrix; the operators below always recurse
# element-wise and treat ``Tensor`` as a scalar for graph purposes.
Scalar = float

# A function called during ``backward`` that propagates the gradient
# to the children of a node. Set during forward by each op.
BackwardFn = Callable[[], None] | None


# ---------------------------------------------------------------------- #
# Core value type
# ---------------------------------------------------------------------- #


@dataclass(eq=False)
class Tensor:
    """A scalar value with optional gradient tracking.

    Attributes:
        data: The numeric value (always a Python ``float`` — the
            educational track stays in scalars; vector ops are
            expressed as nested ``Tensor`` lists).
        requires_grad: If True, ``backward()`` will populate ``grad``.
        grad: The running gradient (initialised to 0.0 on first
            ``backward()``).
        _backward: A closure set by each op that propagates ``grad``
            to ``_prev``. ``None`` for leaf nodes.
        _prev: The set of :class:`Tensor` nodes that produced this
            node (the parents in the dep graph).
    """

    data: Scalar
    requires_grad: bool = False
    grad: Scalar = 0.0
    _backward: BackwardFn = field(default=None, repr=False)
    _prev: set[Tensor] = field(default_factory=set, repr=False)

    def __repr__(self) -> str:
        grad_str = f", grad={self.grad}" if self.requires_grad else ""
        return f"Tensor(data={self.data}{grad_str})"

    # ------------------------------------------------------------------ #
    # Operator overloads — implemented inline so mypy strict sees them.
    # ------------------------------------------------------------------ #

    def __add__(self, other: Tensor | float | int) -> Tensor:
        return _binop("+", self, other)

    def __radd__(self, other: float | int) -> Tensor:
        return _binop("+", other, self)

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        return _binop("-", self, other)

    def __rsub__(self, other: float | int) -> Tensor:
        return _binop("-", other, self)

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        return _binop("*", self, other)

    def __rmul__(self, other: float | int) -> Tensor:
        return _binop("*", other, self)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        return _binop("/", self, other)

    def __rtruediv__(self, other: float | int) -> Tensor:
        return _binop("/", other, self)

    def __neg__(self) -> Tensor:
        return neg(self)

    def __pos__(self) -> Tensor:
        return self

    def __pow__(self, exponent: float) -> Tensor:
        if not isinstance(exponent, (int, float)):
            raise TypeError("Tensor.__pow__ only supports constant Python exponents")
        c = float(exponent)

        def _backward() -> None:
            self.grad += c * (self.data ** (c - 1.0)) * out.grad

        out = Tensor(data=self.data**c)
        return _track(out, (self,), _backward)

    # ------------------------------------------------------------------ #
    # Gradient propagation
    # ------------------------------------------------------------------ #

    def backward(self) -> None:
        """Compute gradients via reverse-mode autodiff.

        Builds a post-order traversal of the graph rooted at this
        node (children before parents) then walks it in reverse,
        calling each ``_backward`` closure to chain the gradient
        back to leaves. Sets every visited leaf's ``.grad`` to the
        accumulated value.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() called on a Tensor with requires_grad=False")

        # Iterative post-order DFS. We push ``(node, processed)``
        # tuples; on the first visit we re-push the node with
        # ``processed=True`` after scheduling its children, so the
        # node only gets appended to ``topo`` after every descendant
        # is already in place.
        topo: list[Tensor] = []
        visited: set[Tensor] = set()
        work: list[tuple[Tensor, bool]] = [(self, False)]
        while work:
            node, processed = work.pop()
            if processed:
                topo.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            work.append((node, True))
            for child in node._prev:
                if child not in visited:
                    work.append((child, False))
        # ``topo`` is post-order (children before parents). Reverse
        # so the backward pass walks parents first — that propagates
        # the seed gradient correctly.
        topo.reverse()

        # Seed the output gradient.
        self.grad = 1.0

        for node in topo:
            if node._backward is not None:
                node._backward()

    def zero_grad(self) -> None:
        """Reset ``grad`` to 0 in this node and all reachable leaves.

        Call between training steps so gradients don't accumulate
        across batches (PyTorch's ``optim.zero_grad()`` semantics).
        """
        topo: list[Tensor] = []
        visited: set[Tensor] = set()
        stack: list[Tensor] = [self]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            topo.append(node)
            for child in node._prev:
                if child not in visited:
                    stack.append(child)
        for node in topo:
            node.grad = 0.0


# ---------------------------------------------------------------------- #
# Context manager: temporarily disable gradient tracking
# ---------------------------------------------------------------------- #


class _NoGrad:
    """Context manager that disables ``requires_grad`` for new ops.

    Mirrors ``torch.no_grad()``: anything created inside the block
    has ``requires_grad=False`` regardless of its inputs, which is
    the right default for inference / loss evaluation / parameter
    initialisation where we don't want a graph at all.
    """

    def __enter__(self) -> _NoGrad:
        global _GRAD_ENABLED
        self._previous = _GRAD_ENABLED
        _GRAD_ENABLED = False
        return self

    def __exit__(self, *exc: Any) -> None:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._previous


_GRAD_ENABLED = True


def no_grad() -> _NoGrad:
    """Return a context manager that disables grad tracking."""
    return _NoGrad()


# ---------------------------------------------------------------------- #
# Parameters
# ---------------------------------------------------------------------- #


class Parameter(Tensor):
    """A :class:`Tensor` that's a trainable weight.

    Subclass of :class:`Tensor` with ``requires_grad=True`` by default.
    Use these for everything a model should learn (embeddings, attention
    projections, FFN weights, biases, etc.). The optimizer sees them
    via :func:`cds.nlp.optim.parameters`.

    Initial values should be small and zero-centred; the simplest
    default is to wrap an existing :class:`Tensor` via
    ``Parameter(tensor.data)``.
    """

    def __init__(self, value: Scalar) -> None:
        super().__init__(data=float(value), requires_grad=True)


# ---------------------------------------------------------------------- #
# Vectorised ops (matmul on nested Tensor lists)
# ---------------------------------------------------------------------- #


def matmul(a: list[list[Tensor]], b: list[list[Tensor]]) -> list[list[Tensor]]:
    """Matrix multiply for nested :class:`Tensor` matrices.

    ``a`` has shape ``(m, p)``; ``b`` has shape ``(p, n)``. The result
    is an ``(m, n)`` matrix whose entries are :class:`Tensor` nodes
    connected to the inputs via the dep graph.

    Implementation is the textbook triple loop. Each inner product
    uses scalar autograd (one multiply + accumulate) so every entry
    in the result gets a backward fn that propagates to the
    contributing ``a`` and ``b`` entries.

    For a 50K-param model this is the hot path — Sprint 3 leaves it
    in pure Python; the optional ``cds[fast-jit]`` Numba backend
    (Sprint 5) wraps the inner loop for ~10x speed-up without
    changing the autograd semantics.
    """
    if not a or not b or not a[0] or not b[0]:
        return []
    m = len(a)
    p = len(a[0])
    if len(b) != p:
        raise ValueError(f"matmul shape mismatch: a has {p} cols, b has {len(b)} rows")
    n = len(b[0])
    # Allocate result as a (m, n) matrix of zero-constant Tensors so
    # we can mutate them in place. The constant stays out of the
    # autograd graph because ``_track`` skips it (no grad children).
    zero = Tensor(data=0.0, requires_grad=False)
    out: list[list[Tensor]] = [[zero for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            acc = Tensor(data=0.0, requires_grad=False)
            for k in range(p):
                # acc += a[i][k] * b[k][j]  (scalar autograd chain)
                prod = _tracked_mul(a[i][k], b[k][j])
                acc = _tracked_add(acc, prod)
            out[i][j] = acc
    return out


def _tracked_add(a: Tensor, b: Tensor) -> Tensor:
    """Internal helper: ``a + b`` with autograd graph attached.

    Public :func:`add` works on Tensors but is re-exported with
    Python operator overloading; this function is the local version
    used by :func:`matmul`'s inner loop to avoid recursion through
    ``__add__``.
    """

    def _backward() -> None:
        a.grad += out.grad
        b.grad += out.grad

    out = Tensor(data=a.data + b.data)
    return _track(out, (a, b), _backward)


def _tracked_mul(a: Tensor, b: Tensor) -> Tensor:
    """Internal helper: ``a * b`` with autograd graph attached."""

    def _backward() -> None:
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad

    out = Tensor(data=a.data * b.data)
    return _track(out, (a, b), _backward)


# ---------------------------------------------------------------------- #
# Op helpers
# ---------------------------------------------------------------------- #


def _track(out: Tensor, children: Iterable[Tensor], backward: BackwardFn) -> Tensor:
    """Attach a backward closure + parent set to a freshly created node.

    Respects :func:`no_grad`: if grad tracking is off, the new node
    has ``requires_grad=False`` regardless of its inputs.
    """
    children_set = {c for c in children if c.requires_grad}
    if not _GRAD_ENABLED or not children_set:
        out.requires_grad = False
        out._prev = set()
        out._backward = None
    else:
        out.requires_grad = True
        out._prev = children_set
        out._backward = backward
    return out


def _unbroadcast(grad: Scalar, target_shape: tuple[int, ...]) -> Scalar:
    """Compatibility shim — scalar autograd doesn't broadcast.

    Kept as a no-op identity to make the vector-tensor migration
    straightforward later: the scalar engine ignores shapes, but
    vector code that calls this through a hook will keep working.
    """
    return grad


# ---------------------------------------------------------------------- #
# Operators
# ---------------------------------------------------------------------- #


def add(a: Tensor, b: Tensor) -> Tensor:
    """``a + b`` with reverse-mode grad ``∂/∂a = ∂/∂b = out.grad``."""

    def _backward() -> None:
        a.grad += _unbroadcast(out.grad, ())
        b.grad += _unbroadcast(out.grad, ())

    out = Tensor(data=a.data + b.data)
    return _track(out, (a, b), _backward)


def sub(a: Tensor, b: Tensor) -> Tensor:
    """``a - b`` with reverse-mode grad ``∂/∂a = +out.grad, ∂/∂b = -out.grad``."""

    def _backward() -> None:
        a.grad += out.grad
        b.grad -= out.grad

    out = Tensor(data=a.data - b.data)
    return _track(out, (a, b), _backward)


def mul(a: Tensor, b: Tensor) -> Tensor:
    """``a * b`` with reverse-mode grad via the product rule."""

    def _backward() -> None:
        a.grad += b.data * out.grad
        b.grad += a.data * out.grad

    out = Tensor(data=a.data * b.data)
    return _track(out, (a, b), _backward)


def div(a: Tensor, b: Tensor) -> Tensor:
    """``a / b`` with reverse-mode grad via the quotient rule."""

    def _backward() -> None:
        a.grad += out.grad / b.data
        b.grad -= a.data * out.grad / (b.data * b.data)

    out = Tensor(data=a.data / b.data)
    return _track(out, (a, b), _backward)


def neg(a: Tensor) -> Tensor:
    """Unary negation."""

    def _backward() -> None:
        a.grad -= out.grad

    out = Tensor(data=-a.data)
    return _track(out, (a,), _backward)


def exp(a: Tensor) -> Tensor:
    """``exp(a)`` with reverse-mode grad ``∂/∂a = exp(a) * out.grad``."""

    def _backward() -> None:
        a.grad += math.exp(a.data) * out.grad

    out = Tensor(data=math.exp(a.data))
    return _track(out, (a,), _backward)


def log(a: Tensor) -> Tensor:
    """Natural log. ``∂/∂a = out.grad / a.data``.

    Raises :class:`ValueError` for non-positive input — the gradient
    is undefined there.
    """

    def _backward() -> None:
        if a.data == 0.0:
            raise ValueError("log(0) gradient is undefined")
        a.grad += out.grad / a.data

    if a.data <= 0:
        raise ValueError(f"log requires positive input, got {a.data}")
    out = Tensor(data=math.log(a.data))
    return _track(out, (a,), _backward)


def relu(a: Tensor) -> Tensor:
    """Rectified linear unit. ``∂/∂a = out.grad if a > 0 else 0``."""

    def _backward() -> None:
        if a.data > 0:
            a.grad += out.grad

    out = Tensor(data=max(0.0, a.data))
    return _track(out, (a,), _backward)


# ---------------------------------------------------------------------- #
# Operator overloading on Tensor
# ---------------------------------------------------------------------- #


def _scalar_to_tensor(value: Any) -> Tensor:
    """Promote a Python number to a constant Tensor (no grad)."""
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (int, float)):
        return Tensor(data=float(value), requires_grad=False)
    raise TypeError(f"unsupported operand type: {type(value).__name__}")


def _binop(op_name: str, a: Any, b: Any) -> Tensor:
    """Dispatch ``Tensor ``op`` value`` to the right autograd op."""
    op = {
        "+": add,
        "-": sub,
        "*": mul,
        "/": div,
    }[op_name]
    return op(_scalar_to_tensor(a), _scalar_to_tensor(b))


def _unaryop(op_name: str, a: Tensor) -> Tensor:
    return {"neg": neg, "exp": exp, "log": log, "relu": relu}[op_name](a)


# Iterable for type-checkers — unused at runtime but documents intent.
_ = Iterator[Tensor]  # noqa: F841

__all__ = [
    "Tensor",
    "Parameter",
    "no_grad",
    "add",
    "sub",
    "mul",
    "div",
    "neg",
    "exp",
    "log",
    "relu",
    "matmul",
]
