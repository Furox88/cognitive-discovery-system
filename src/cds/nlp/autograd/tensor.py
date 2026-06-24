"""The core :class:`Tensor` graph node and the ops coupled to it.

The elementwise arithmetic ops (:func:`add`, :func:`sub`, :func:`mul`,
:func:`div`, :func:`neg`) live here rather than in :mod:`cds.nlp.autograd.ops`
because :class:`Tensor`'s dunder operators (``__add__`` etc.) dispatch
to them directly via :func:`_binop`. Keeping them in the same module
avoids a ``tensor <-> ops`` import cycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cds.nlp.autograd._grad import BackwardFn, Scalar, _track

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
        """Return ``Tensor(data=..., grad=...)`` (``grad`` only when tracking)."""
        grad_str = f", grad={self.grad}" if self.requires_grad else ""
        return f"Tensor(data={self.data}{grad_str})"

    # ------------------------------------------------------------------ #
    # Operator overloads — implemented inline so mypy strict sees them.
    # Each dispatches to :func:`_binop` (or :func:`neg`) so a graph node
    # carrying a ``_backward`` closure is returned and grads flow.
    # ------------------------------------------------------------------ #

    def __add__(self, other: Tensor | float | int) -> Tensor:
        """Return ``self + other`` as a tracked sum node."""
        return _binop("+", self, other)

    def __radd__(self, other: float | int) -> Tensor:
        """Return ``other + self`` (``other`` is a number) as a tracked sum node."""
        return _binop("+", other, self)

    def __sub__(self, other: Tensor | float | int) -> Tensor:
        """Return ``self - other`` as a tracked difference node."""
        return _binop("-", self, other)

    def __rsub__(self, other: float | int) -> Tensor:
        """Return ``other - self`` (``other`` is a number) as a tracked difference node."""
        return _binop("-", other, self)

    def __mul__(self, other: Tensor | float | int) -> Tensor:
        """Return ``self * other`` as a tracked product node."""
        return _binop("*", self, other)

    def __rmul__(self, other: float | int) -> Tensor:
        """Return ``other * self`` (``other`` is a number) as a tracked product node."""
        return _binop("*", other, self)

    def __truediv__(self, other: Tensor | float | int) -> Tensor:
        """Return ``self / other`` as a tracked quotient node."""
        return _binop("/", self, other)

    def __rtruediv__(self, other: float | int) -> Tensor:
        """Return ``other / self`` (``other`` is a number) as a tracked quotient node."""
        return _binop("/", other, self)

    def __neg__(self) -> Tensor:
        """Return the negation ``-self`` as a tracked node."""
        return neg(self)

    def __pos__(self) -> Tensor:
        """Return ``+self`` (a no-op copy of this node)."""
        return self

    def __pow__(self, exponent: float) -> Tensor:
        """Return ``self ** exponent`` (constant power) as a tracked node."""
        # Return ``NotImplemented`` for unsupported operand types instead of
        # raising — this is the Pythonic contract for arithmetic dunders
        # (lets Python try the reflected ``__rpow__`` and only raise a real
        # ``TypeError`` if neither side can handle it). CodeQL's
        # ``unexpected-raise-in-special-method`` flags ``raise`` in dunders
        # precisely because it short-circuits that reflection protocol.
        if not isinstance(exponent, (int, float)):
            # Returning NotImplemented is correct here even though the declared
            # return type is Tensor: CPython's binary-operator dispatch
            # consumes the value (it never reaches user code), and mypy
            # models NotImplemented as compatible with arithmetic-dunder
            # return types for exactly this reason.
            return NotImplemented
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
            # Defensive duplicate-pop guard. Unreachable given the LIFO
            # stack + the ``child not in visited`` filter below: that pair
            # mathematically prevents any node from being pushed twice, so
            # the re-pop never happens. Kept to mirror ``backward()``'s
            # defensive structure and to stay robust if the push filter
            # is ever relaxed.
            if node in visited:  # pragma: no cover
                continue
            visited.add(node)
            topo.append(node)
            for child in node._prev:
                if child not in visited:
                    stack.append(child)
        for node in topo:
            node.grad = 0.0


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
        """Store ``value`` as a leaf with ``requires_grad=True``."""
        super().__init__(data=float(value), requires_grad=True)


# ---------------------------------------------------------------------- #
# Elementwise arithmetic ops (coupled to Tensor's dunders via _binop)
# ---------------------------------------------------------------------- #


def add(a: Tensor, b: Tensor) -> Tensor:
    """``a + b`` with reverse-mode grad ``∂/∂a = ∂/∂b = out.grad``."""

    def _backward() -> None:
        a.grad += out.grad
        b.grad += out.grad

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


# ---------------------------------------------------------------------- #
# Operator overloading glue
# ---------------------------------------------------------------------- #


def _scalar_to_tensor(value: Tensor | int | float) -> Tensor:
    """Promote a Python number to a constant Tensor (no grad)."""
    if isinstance(value, Tensor):
        return value
    if isinstance(value, (int, float)):
        return Tensor(data=float(value), requires_grad=False)
    raise TypeError(f"unsupported operand type: {type(value).__name__}")


def _binop(op_name: str, a: Tensor | int | float, b: Tensor | int | float) -> Tensor:
    """Dispatch ``Tensor ``op`` value`` to the right autograd op."""
    op = {
        "+": add,
        "-": sub,
        "*": mul,
        "/": div,
    }[op_name]
    return op(_scalar_to_tensor(a), _scalar_to_tensor(b))
