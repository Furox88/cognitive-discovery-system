"""Transformer building blocks: GeLU, LayerNorm, FFN, transformer block.

Provides the layer-level pieces that wrap a multi-head attention call
into a proper Transformer block, following the pre-norm convention
from the original paper (Vaswani et al. 2017 §3.1) but with the
post-norm variant available for completeness.

A transformer block is::

    y = x + MHSA(LN(x))
    z = y + FFN(LN(y))

Where ``MHSA`` is :func:`cds.nlp.attention.multi_head_attention` and
``FFN`` is the two-layer position-wise feed-forward network with a
GeLU non-linearity.

Scope:
- Pre-norm and post-norm block variants.
- GeLU (exact, via ``math.erf``).
- LayerNorm with learnable ``gamma`` / ``beta`` scale and shift.
- Position-wise feed-forward with one hidden expansion (typically
  ``d_ff = 4 * d_model``).

Scope explicitly *out* for Sprint 2:
- Dropout, residual scaling, GQA / MQA, ALiBi, RoPE — all useful
  modern variants; tracked for a later educational add-on if the
  attention block is fast enough to use in demos.
- Training-time autograd. The block is functional at inference
  time; gradients land in Sprint 3.
"""

from __future__ import annotations

import math
from typing import TypedDict

from cds.nlp.attention import matmul, multi_head_attention

__all__ = [
    "gelu",
    "layer_norm",
    "feed_forward",
    "transformer_block",
]


class AttentionWeights(TypedDict):
    """Weight dict for the multi-head attention sub-block."""

    w_q: list[list[float]]
    w_k: list[list[float]]
    w_v: list[list[float]]
    w_o: list[list[float]]
    ln1_gamma: list[float]
    ln1_beta: list[float]
    ln2_gamma: list[float]
    ln2_beta: list[float]


class FeedForwardWeights(TypedDict):
    """Weight dict for the position-wise feed-forward sub-block."""

    w1: list[list[float]]
    b1: list[float]
    w2: list[list[float]]
    b2: list[float]


# ---------------------------------------------------------------------- #
# Activations & normalisation
# ---------------------------------------------------------------------- #


def gelu(x: float) -> float:
    """Exact Gaussian Error Linear Unit activation.

    ``GELU(x) = x * Phi(x)`` where ``Phi`` is the standard normal CDF.
    Computed via ``0.5 * x * (1 + erf(x / sqrt(2)))`` for numerical
    accuracy. The Tanh approximation used in some papers
    (``0.5 x (1 + tanh(...))``) is faster but introduces a small bias
    that doesn't matter for educational use — the exact form costs
    nothing here.
    """
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


def layer_norm(
    x: list[list[float]],
    gamma: list[float],
    beta: list[float],
    eps: float = 1e-5,
) -> list[list[float]]:
    """Layer normalisation over the last dimension.

    For each row of ``x``:
        mean = E[x]
        var  = E[(x - mean)^2]
        y    = gamma * (x - mean) / sqrt(var + eps) + beta

    Args:
        x: Input, shape ``(n, d)``.
        gamma: Per-feature scale, length ``d``.
        beta: Per-feature shift, length ``d``.
        eps: Variance floor for numerical stability.
    """
    if not x:
        return []
    d = len(x[0])
    if len(gamma) != d or len(beta) != d:
        raise ValueError(f"gamma/beta length {len(gamma)}/{len(beta)} != feature dim {d}")
    out: list[list[float]] = []
    for row in x:
        inv_d = 1.0 / d
        mean = sum(row) * inv_d
        var = sum((xi - mean) ** 2 for xi in row) * inv_d
        std = math.sqrt(var + eps)
        out.append([gamma[j] * (row[j] - mean) / std + beta[j] for j in range(d)])
    return out


# ---------------------------------------------------------------------- #
# Feed-forward network
# ---------------------------------------------------------------------- #


def feed_forward(
    x: list[list[float]],
    w1: list[list[float]],
    b1: list[float],
    w2: list[list[float]],
    b2: list[float],
) -> list[list[float]]:
    """Two-layer position-wise FFN with GeLU.

    ``FFN(x) = (GeLU(x W1 + b1)) W2 + b2``

    Args:
        x: Input, shape ``(n, d_model)``.
        w1: First weight matrix, shape ``(d_model, d_ff)``.
        b1: First bias, length ``d_ff``.
        w2: Second weight matrix, shape ``(d_ff, d_model)``.
        b2: Second bias, length ``d_model``.
    """
    if not x:
        return []
    d_model = len(x[0])
    if len(b2) != d_model:
        raise ValueError(f"b2 length {len(b2)} != d_model {d_model}")
    d_ff = len(b1)
    if len(w1) != d_model or len(w1[0]) != d_ff:
        raise ValueError(
            f"w1 shape {len(w1)}x{len(w1[0]) if w1 else 0} != expected ({d_model}, {d_ff})"
        )

    h = matmul(x, w1)
    for i, row in enumerate(h):
        for j in range(d_ff):
            row[j] = gelu(row[j] + b1[j])
    out = matmul(h, w2)
    for i, row in enumerate(out):
        for j in range(d_model):
            row[j] = row[j] + b2[j]
    return out


# ---------------------------------------------------------------------- #
# Transformer block
# ---------------------------------------------------------------------- #


def _add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Element-wise add of two ``(n, d)`` matrices (defensive copy)."""
    if not a:
        return []
    n = len(a)
    d = len(a[0])
    if len(b) != n or len(b[0]) != d:
        raise ValueError(f"add shape mismatch: {n}x{d} vs {len(b)}x{len(b[0]) if b else 0}")
    return [[a[i][j] + b[i][j] for j in range(d)] for i in range(n)]


def transformer_block(
    x: list[list[float]],
    attn_weights: AttentionWeights,
    ffn_weights: FeedForwardWeights,
    n_heads: int,
    mask: list[list[float]] | None = None,
    prenorm: bool = True,
) -> list[list[float]]:
    """One Transformer encoder block.

    Args:
        x: Input sequence, shape ``(n, d_model)``.
        attn_weights: :class:`AttentionWeights` dict.
        ffn_weights: :class:`FeedForwardWeights` dict.
        n_heads: Number of attention heads.
        mask: Optional additive attention mask.
        prenorm: If True (default), apply LayerNorm *before* attention
            and FFN (Vaswani 2017 default in modern code; the paper
            used post-norm, but prenorm trains more stably).

    Returns:
        Output sequence, shape ``(n, d_model)``.
    """
    if not x:
        return []
    # TypedDict gives us a precise type for each key — no narrowing
    # or cast needed for the LayerNorm parameters.
    ln1_gamma = attn_weights["ln1_gamma"]
    ln1_beta = attn_weights["ln1_beta"]
    ln2_gamma = attn_weights["ln2_gamma"]
    ln2_beta = attn_weights["ln2_beta"]

    if prenorm:
        normed = layer_norm(x, ln1_gamma, ln1_beta)
        attn_out = multi_head_attention(
            normed,
            attn_weights["w_q"],
            attn_weights["w_k"],
            attn_weights["w_v"],
            attn_weights["w_o"],
            n_heads,
            mask,
        )
        x = _add(x, attn_out)

        normed2 = layer_norm(x, ln2_gamma, ln2_beta)
        ffn_out = feed_forward(
            normed2,
            ffn_weights["w1"],
            ffn_weights["b1"],
            ffn_weights["w2"],
            ffn_weights["b2"],
        )
        x = _add(x, ffn_out)
        return x

    # Post-norm: original paper convention. Less stable for deep stacks
    # but kept for completeness in the educational track.
    attn_out = multi_head_attention(
        x,
        attn_weights["w_q"],
        attn_weights["w_k"],
        attn_weights["w_v"],
        attn_weights["w_o"],
        n_heads,
        mask,
    )
    x = layer_norm(_add(x, attn_out), ln1_gamma, ln1_beta)
    ffn_out = feed_forward(
        x,
        ffn_weights["w1"],
        ffn_weights["b1"],
        ffn_weights["w2"],
        ffn_weights["b2"],
    )
    x = layer_norm(_add(x, ffn_out), ln2_gamma, ln2_beta)
    return x
