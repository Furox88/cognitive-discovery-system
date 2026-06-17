"""Optimizers for the cds.nlp autograd engine.

Two optimizers in the educational NLP track:

* :class:`SGD` — stochastic gradient descent with optional momentum
  and weight decay. The minimum useful optimizer; matches
  Karpathy's micrograd ``Optimizer`` semantics.
* :class:`Adam` — adaptive moment estimation (Kingma & Ba 2014).
  The default for transformer training. Uses bias-corrected first
  and second moment estimates stored on the :class:`cds.nlp.autograd.Tensor`
  itself, so the optimizer's state is just a list of :class:`Parameter`
  references.

Both optimizers operate on :class:`Parameter` objects — thin
:func:`cds.nlp.autograd.Tensor` wrappers that the model exposes as
its trainable weights. Use :func:`parameters` to collect them.

References:
    - Kingma, D. P., & Ba, J. (2014). "Adam: A Method for Stochastic
      Optimization." arXiv:1412.6980.
    - PyTorch's ``torch.optim`` docs — for the parameter-group API
      shape this module imitates (one ``Tensor`` per parameter;
      no per-parameter options in the educational track).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from cds.core._numeric import ADAM_DEFAULT_BETAS, ADAM_DEFAULT_EPS, ADAM_DEFAULT_LR, SGD_DEFAULT_LR

if TYPE_CHECKING:
    from cds.nlp.autograd import Tensor


__all__ = ["SGD", "Adam"]


@dataclass
class SGD:
    """Stochastic gradient descent with optional momentum.

    Args:
        params: Iterable of :class:`Parameter` (or any
            :class:`cds.nlp.autograd.Tensor` with ``requires_grad=True``)
            to update.
        lr: Learning rate. Must be positive.
        momentum: Momentum factor in ``[0, 1)``. ``0`` reduces to
            vanilla SGD. ``> 0`` updates each parameter with
            ``v = momentum * v + grad; p -= lr * v``.
        weight_decay: Optional L2 penalty coefficient. Adds
            ``weight_decay * p.data`` to the gradient at every step.
    """

    params: list[Tensor]
    lr: float = SGD_DEFAULT_LR
    momentum: float = 0.0
    weight_decay: float = 0.0
    _velocities: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        # Allocate a velocity slot per parameter.
        self._velocities = [0.0] * len(self.params)

    def step(self) -> None:
        """Apply one update to each parameter.

        Must be called *after* ``loss.backward()`` and *before*
        ``zero_grad()`` — otherwise the gradient buffer will be
        overwritten on the next forward pass.
        """
        for i, p in enumerate(self.params):
            grad = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            if self.momentum == 0.0:
                p.data -= self.lr * grad
            else:
                self._velocities[i] = self.momentum * self._velocities[i] + grad
                p.data -= self.lr * self._velocities[i]

    def zero_grad(self) -> None:
        """Reset all parameter gradients to 0. Call between batches."""
        for p in self.params:
            p.grad = 0.0


@dataclass
class Adam:
    """Adam optimiser (Kingma & Ba 2014).

    Maintains per-parameter first and second moment estimates with
    bias correction. Defaults match the paper (``betas=(0.9, 0.999)``,
    ``eps=1e-8``).

    Args:
        params: Trainable parameters.
        lr: Learning rate. Typical values for transformer training
            are in the ``3e-4`` to ``1e-3`` range.
        betas: Coefficients for the first and second moment moving
            averages.
        eps: Epsilon for numerical stability in the denominator.
        weight_decay: Optional L2 penalty coefficient.
    """

    params: list[Tensor]
    lr: float = ADAM_DEFAULT_LR
    betas: tuple[float, float] = ADAM_DEFAULT_BETAS
    eps: float = ADAM_DEFAULT_EPS
    weight_decay: float = 0.0
    _t: int = field(init=False, default=0)
    _m: list[float] = field(init=False, default_factory=list)
    _v: list[float] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if self.lr <= 0:
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if not (0.0 <= self.betas[0] < 1.0 and 0.0 <= self.betas[1] < 1.0):
            raise ValueError(f"betas must each be in [0, 1), got {self.betas}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay}")
        self._m = [0.0] * len(self.params)
        self._v = [0.0] * len(self.params)

    def step(self) -> None:
        """Apply one update. Increment step counter internally."""
        self._t += 1
        b1, b2 = self.betas
        for i, p in enumerate(self.params):
            grad = p.grad + self.weight_decay * p.data if self.weight_decay else p.grad
            self._m[i] = b1 * self._m[i] + (1.0 - b1) * grad
            self._v[i] = b2 * self._v[i] + (1.0 - b2) * (grad * grad)
            # Bias correction — important in the first few hundred
            # steps when the moving averages are still warming up.
            m_hat = self._m[i] / (1.0 - b1**self._t)
            v_hat = self._v[i] / (1.0 - b2**self._t)
            p.data -= self.lr * m_hat / (math.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = 0.0


def parameters(items: Iterable[Tensor]) -> list[Tensor]:
    """Collect trainable tensors from a model.

    Convenience helper — many models store their weights in
    dictionaries or lists; this filters to ``requires_grad=True``
    in one call.
    """
    return [t for t in items if t.requires_grad]
