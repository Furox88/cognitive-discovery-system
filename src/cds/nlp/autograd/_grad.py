"""Gradient-tracking context and shared graph helpers.

The single shared piece of mutable state in the autograd engine is the
module global :data:`_GRAD_ENABLED`, flipped by :class:`_NoGrad` and
read by :func:`_track`. Keeping these symbols in their own leaf module
lets :mod:`cds.nlp.autograd.tensor` and :mod:`cds.nlp.autograd.ops`
both depend on them without creating an import cycle.

This module deliberately does **not** import :class:`~cds.nlp.autograd.tensor.Tensor`
— not even under ``TYPE_CHECKING`` — because that would reintroduce the
``tensor <-> _grad`` cycle. :func:`_track` is typed against a local
:class:`_Trackable` protocol (structural typing), so the dependency graph
stays a strict DAG: ``_grad`` is the leaf both ``tensor`` and ``ops``
point at, with no edge back.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class _Trackable(Protocol):
    """Structural supertype of :class:`~cds.nlp.autograd.tensor.Tensor`.

    Defines only the attributes :func:`_track` mutates, so the grad helper
    can depend on a shape rather than the concrete ``Tensor`` class and the
    ``tensor -> _grad -> tensor`` import cycle is broken.
    """

    requires_grad: bool
    _prev: set[Any]
    _backward: Callable[[], None] | None


# ``_T`` is bound to :class:`_Trackable` so :func:`_track` is parametric:
# whatever the caller passes in (in practice always :class:`Tensor`, but
# the grad module can't name that type without re-importing it and
# reintroducing the cycle) is the type returned. This is what lets the
# op implementations stay typed as ``Tensor -> Tensor`` even though
# ``_grad`` only knows the structural :class:`_Trackable` shape.
_T = TypeVar("_T", bound=_Trackable)

# ---------------------------------------------------------------------- #
# Type aliases
# ---------------------------------------------------------------------- #

# The underlying scalar type. Pure Python floats throughout — the
# autograd engine doesn't care whether a Tensor represents a single
# float or a (deferred) matrix; the operators always recurse
# element-wise and treat ``Tensor`` as a scalar for graph purposes.
Scalar = float

# A function called during ``backward`` that propagates the gradient
# to the children of a node. Set during forward by each op.
BackwardFn = Callable[[], None] | None


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

    def __exit__(self, *exc: object) -> None:
        global _GRAD_ENABLED
        _GRAD_ENABLED = self._previous


_GRAD_ENABLED = True


def no_grad() -> _NoGrad:
    """Return a context manager that disables grad tracking."""
    return _NoGrad()


# ---------------------------------------------------------------------- #
# Shared op helper
# ---------------------------------------------------------------------- #


def _track(out: _T, children: Iterable[_Trackable], backward: BackwardFn) -> _T:
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
