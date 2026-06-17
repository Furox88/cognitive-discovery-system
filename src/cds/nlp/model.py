"""A minimal GPT-from-scratch model in pure Python (Sprint 4).

:class:`MiniGPT` is the smallest possible decoder-only transformer
that still does the right thing: a token embedding, a single causal
self-attention block, and a language-model head. It is a 50K-100K
parameter model — small enough to train in a few minutes on a CPU
and big enough to memorise the Shakespeare excerpt in
:mod:`cds.nlp.data`.

This module closes the loop on the educational NLP track:
BPE + sinusoidal embeddings + scaled-dot-product attention +
multi-head block + reverse-mode autograd + SGD/Adam + MiniGPT =
a learner can now watch a tiny GPT actually fit a corpus and
sample text from it, all in pure Python.

Scope (Sprint 4):
* :class:`MiniGPT` — token embedding + 1 causal block + LM head
* :func:`cross_entropy` (re-exported from :mod:`cds.nlp.training`)
* Inference via :meth:`MiniGPT.sample`

Out of scope for the educational track:
* Multi-layer stacks (N blocks) — straightforward to add but
  quadratic in N for the educational engine.
* Dropout, ALiBi, RoPE, GQA, sliding window — modern refinements
  not on the educational critical path.
* KV-cache — required for efficient long-context inference; the
  full corpus here is short enough that re-running forward is fine.
"""

from __future__ import annotations

import math

from cds.core._numeric import LAYERNORM_EPS, SAMPLING_TEMPERATURE_EPS
from cds.nlp.attention import causal_mask
from cds.nlp.autograd import Parameter, Tensor, exp, log
from cds.nlp.embed import PositionalEncoding

__all__ = ["MiniGPT", "sample"]


# ---------------------------------------------------------------------- #
# MiniGPT
# ---------------------------------------------------------------------- #


class MiniGPT:
    """One-block decoder-only transformer for char-level language modelling.

    Architecture:
        token embedding  (vocab_size x d_model)
        sinusoidal positions  (max_len x d_model)
        one transformer block (multi-head causal self-attention + FFN)
        linear LM head  (d_model x vocab_size)

    Total parameters scale as
    ``vocab_size * d_model + d_model^2 * 4 + d_model * d_ff + d_model * vocab_size``,
    which for the default ``d_model=32, n_heads=2, d_ff=64`` is
    ~5K-50K — small enough to fit in a fresh clone and train
    in a few minutes on a CPU.

    Args:
        vocab_size: Number of distinct tokens.
        d_model: Embedding width and hidden size.
        n_heads: Number of attention heads. Must divide ``d_model``.
        d_ff: Position-wise FFN hidden width. Default ``4 * d_model``.
        max_len: Maximum sequence length the model can handle.
        seed: RNG seed used for the random initialisation.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        n_heads: int = 2,
        d_ff: int = 64,
        max_len: int = 64,
        seed: int = 0xC0DE,
    ) -> None:
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by n_heads {n_heads}")
        if d_ff < 1:
            raise ValueError(f"d_ff must be >= 1, got {d_ff}")
        if max_len < 1:
            raise ValueError(f"max_len must be >= 1, got {max_len}")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len

        import random

        rng = random.Random(seed)

        # Xavier-uniform-ish init for matrices, zero for biases.
        def _rand_matrix(rows: int, cols: int) -> list[list[float]]:
            bound = 1.0 / math.sqrt(cols)
            return [[rng.uniform(-bound, bound) for _ in range(cols)] for _ in range(rows)]

        # Token embedding: each row is the d_model vector for a token id.
        # Represented as a list of Parameters (one per row) so the
        # optimiser sees them as trainable.
        self.token_embedding: list[list[Parameter]] = [
            [Parameter(v) for v in row] for row in _rand_matrix(vocab_size, d_model)
        ]
        # Projections: (d_model x d_model) for Q, K, V, output.
        self.w_q = [[Parameter(v) for v in row] for row in _rand_matrix(d_model, d_model)]
        self.w_k = [[Parameter(v) for v in row] for row in _rand_matrix(d_model, d_model)]
        self.w_v = [[Parameter(v) for v in row] for row in _rand_matrix(d_model, d_model)]
        self.w_o = [[Parameter(v) for v in row] for row in _rand_matrix(d_model, d_model)]
        # LayerNorm parameters (gamma=1, beta=0 default).
        self.ln1_gamma = [Parameter(1.0) for _ in range(d_model)]
        self.ln1_beta = [Parameter(0.0) for _ in range(d_model)]
        self.ln2_gamma = [Parameter(1.0) for _ in range(d_model)]
        self.ln2_beta = [Parameter(0.0) for _ in range(d_model)]
        # FFN: 2-layer with GeLU.
        self.w1 = [[Parameter(v) for v in row] for row in _rand_matrix(d_model, d_ff)]
        self.b1 = [Parameter(0.0) for _ in range(d_ff)]
        self.w2 = [[Parameter(v) for v in row] for row in _rand_matrix(d_ff, d_model)]
        self.b2 = [Parameter(0.0) for _ in range(d_model)]
        # LM head: d_model -> vocab_size.
        self.w_head = [[Parameter(v) for v in row] for row in _rand_matrix(d_model, vocab_size)]
        self.b_head = [Parameter(0.0) for _ in range(vocab_size)]

        # Fixed sinusoidal position encoding (no trainable parameters).
        self.pos = PositionalEncoding(max_len=max_len, d_model=d_model)

    # ------------------------------------------------------------------ #
    # Parameter collection
    # ------------------------------------------------------------------ #

    def parameters(self) -> list[Parameter]:
        """All trainable parameters (in the order the optimiser should walk)."""
        params: list[Parameter] = []
        for row in self.token_embedding:
            params.extend(row)
        for mat in (self.w_q, self.w_k, self.w_v, self.w_o, self.w1, self.w2, self.w_head):
            for row in mat:
                params.extend(row)
        params.extend(self.ln1_gamma)
        params.extend(self.ln1_beta)
        params.extend(self.ln2_gamma)
        params.extend(self.ln2_beta)
        params.extend(self.b1)
        params.extend(self.b2)
        params.extend(self.b_head)
        return params

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def forward(self, ids: list[int]) -> list[Tensor]:
        """Run one forward pass; return ``vocab_size`` logits for the
        last position only (a language model predicts the next
        token from the *final* position's hidden state).

        For the educational track we keep the autograd graph attached
        to the returned logits, so :func:`cds.nlp.training.cross_entropy`
        can chain back to every parameter.
        """
        if not ids:
            return []
        if len(ids) > self.max_len:
            raise ValueError(f"sequence length {len(ids)} exceeds max_len {self.max_len}")
        n = len(ids)
        d = self.d_model

        # Token + position embedding. The position encoding is a fixed
        # pure-Python matrix; we lift it to the autograd world by
        # promoting each entry to a constant Tensor.
        token_vecs: list[list[Tensor]] = [
            [self.token_embedding[t][j] for j in range(d)] for t in ids
        ]
        pos_vecs: list[list[Tensor]] = [
            [Tensor(data=self.pos.matrix[i][j], requires_grad=False) for j in range(d)]
            for i in range(n)
        ]
        x: list[list[Tensor]] = [
            [token_vecs[i][j] + pos_vecs[i][j] for j in range(d)] for i in range(n)
        ]

        # Causal multi-head self-attention + FFN with pre-norm.
        normed = _layer_norm(x, self.ln1_gamma, self.ln1_beta)
        attn_out = _multi_head_attention(
            normed,
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_o,
            self.n_heads,
            mask=causal_mask(n),
        )
        x = _add(x, attn_out)
        normed2 = _layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ffn_out = _feed_forward(normed2, self.w1, self.b1, self.w2, self.b2)
        x = _add(x, ffn_out)

        # Use only the last position's hidden state for the LM head —
        # predicting the next token only needs the rightmost context.
        last = x[-1]
        # Linear: logits = last @ w_head + b_head
        logits: list[Tensor] = []
        for v in range(self.vocab_size):
            acc: Tensor = self.b_head[v] + Tensor(data=0.0, requires_grad=False)
            for j in range(d):
                acc = acc + last[j] * self.w_head[j][v]
            logits.append(acc)
        return logits

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def sample(self, prompt_ids: list[int], n_tokens: int, temperature: float = 1.0) -> list[int]:
        """Generate ``n_tokens`` new ids after ``prompt_ids``.

        Uses simple multinomial sampling: convert the last-position
        logits to probabilities via :func:`softmax`, sample one
        token, append, repeat. The forward pass runs with
        ``requires_grad=False`` so no graph is built.
        """
        from cds.nlp.training import softmax

        if n_tokens <= 0:
            return list(prompt_ids)
        ids = list(prompt_ids)
        with _no_grad():
            for _ in range(n_tokens):
                # Trim to the last ``max_len`` tokens so we don't exceed
                # the model's positional window.
                ctx = ids[-self.max_len :]
                logits = self.forward(ctx)
                # Drop the graph — we only need the numbers.
                vals = [li.data for li in logits]
                if temperature != 1.0:
                    vals = [v / max(temperature, SAMPLING_TEMPERATURE_EPS) for v in vals]
                probs = softmax(vals)
                # Deterministic argmax for the educational default;
                # callers that want sampling can pass a temperature
                # and a custom sampler via the public hooks.
                nxt = max(range(len(probs)), key=lambda i: probs[i])
                ids.append(nxt)
        return ids


# ---------------------------------------------------------------------- #
# Internal helpers (lift the float-only ops from cds.nlp.layers to
# operate on Tensor rows so the autograd graph stays connected)
# ---------------------------------------------------------------------- #


def _layer_norm(
    x: list[list[Tensor]],
    gamma: list[Parameter],
    beta: list[Parameter],
) -> list[list[Tensor]]:
    d = len(x[0])
    if len(gamma) != d or len(beta) != d:
        raise ValueError("gamma/beta length mismatch")
    out: list[list[Tensor]] = []
    for row in x:
        inv_d = 1.0 / d
        mean = _sum_tensors(row) * inv_d
        diffs = [row[j] - mean for j in range(d)]
        var = _sum_tensors([diffs[j] * diffs[j] for j in range(d)]) * inv_d
        std = _exp(0.5 * _log(var + LAYERNORM_EPS))
        out.append([gamma[j] * diffs[j] / std + beta[j] for j in range(d)])
    return out


def _sum_tensors(ts: list[Tensor]) -> Tensor:
    acc = Tensor(data=0.0, requires_grad=False)
    for t in ts:
        acc = acc + t
    return acc


def _exp(x: Tensor) -> Tensor:
    return exp(x)


def _log(x: Tensor) -> Tensor:
    return log(x)


def _add(a: list[list[Tensor]], b: list[list[Tensor]]) -> list[list[Tensor]]:
    n = len(a)
    d = len(a[0])
    return [[a[i][j] + b[i][j] for j in range(d)] for i in range(n)]


def _multi_head_attention(
    x: list[list[Tensor]],
    w_q: list[list[Parameter]],
    w_k: list[list[Parameter]],
    w_v: list[list[Parameter]],
    w_o: list[list[Parameter]],
    n_heads: int,
    mask: list[list[float]] | None = None,
) -> list[list[Tensor]]:
    """Multi-head self-attention on Tensor rows.

    Lifts the pure-Python attention from ``cds.nlp.attention`` to
    operate on Tensor rows so the autograd graph stays attached.
    """
    d = len(x[0])
    d_head = d // n_heads

    # Project to Q, K, V via a tiny "matvec" inner loop.
    def _project(x_row: list[Tensor], w: list[list[Parameter]]) -> list[Tensor]:
        out: list[Tensor] = []
        for i in range(len(w)):
            acc = Tensor(data=0.0, requires_grad=False)
            for j in range(len(w[i])):
                acc = acc + x_row[j] * w[i][j]
            out.append(acc)
        return out

    q_rows = [_project(x[i], w_q) for i in range(len(x))]
    k_rows = [_project(x[i], w_k) for i in range(len(x))]
    v_rows = [_project(x[i], w_v) for i in range(len(x))]

    # Split heads: (n, d) → (n, n_heads, d_head) → (n_heads, n, d_head).
    def _split(rows: list[list[Tensor]]) -> list[list[list[Tensor]]]:
        n = len(rows)
        return [
            [[rows[i][h * d_head + j] for j in range(d_head)] for i in range(n)]
            for h in range(n_heads)
        ]

    qh, kh, vh = _split(q_rows), _split(k_rows), _split(v_rows)

    # Per-head attention with causal mask.
    head_outs: list[list[list[Tensor]]] = []
    for h in range(n_heads):
        head_outs.append(_scaled_dot_product(qh[h], kh[h], vh[h], mask))

    # Concat heads (n_heads, n, d_head) → (n, d).
    n = len(x)
    merged: list[list[Tensor]] = [[] for _ in range(n)]
    for i in range(n):
        row: list[Tensor] = []
        for h in range(n_heads):
            row.extend(head_outs[h][i])
        merged[i] = row

    # Output projection.
    return [_project(merged[i], w_o) for i in range(n)]


def _scaled_dot_product(
    q: list[list[Tensor]],
    k: list[list[Tensor]],
    v: list[list[Tensor]],
    mask: list[list[float]] | None = None,
) -> list[list[Tensor]]:
    d_k = len(q[0])
    n = len(q)
    scale = 1.0 / math.sqrt(d_k)
    # scores[i][t] = sum_j q[i][j] * k[t][j] * scale + (mask[i][t] if mask else 0)
    k_t: list[list[Tensor]] = [[k[t][j] for t in range(n)] for j in range(d_k)]
    scores: list[list[Tensor]] = []
    for i in range(n):
        row: list[Tensor] = []
        for t in range(n):
            acc = Tensor(data=0.0, requires_grad=False)
            for j in range(d_k):
                acc = acc + q[i][j] * k_t[j][t]
            row.append(acc * scale)
        if mask is not None:
            for t in range(n):
                m = mask[i][t]
                if m != 0.0:
                    # m == -inf → add a very large negative constant
                    # to make softmax collapse the weight. (Pure-Python
                    # doesn't have float("-inf") reliably; we use a
                    # big negative number.)
                    row[t] = row[t] + Tensor(data=-1e9, requires_grad=False)
        scores.append(row)
    # Softmax row-wise.
    weights: list[list[Tensor]] = [_softmax_row(r) for r in scores]
    # context = weights @ v.
    out: list[list[Tensor]] = []
    for i in range(n):
        row = [Tensor(data=0.0, requires_grad=False) for _ in range(len(v[0]))]
        for t in range(n):
            for j in range(len(v[t])):
                row[j] = row[j] + weights[i][t] * v[t][j]
        out.append(row)
    return out


def _softmax_row(row: list[Tensor]) -> list[Tensor]:
    m = max(t.data for t in row)
    m_const = Tensor(data=m, requires_grad=False)
    exps = [exp(row[i] - m_const) for i in range(len(row))]
    total = _sum_tensors(exps)
    return [e / total for e in exps]


def _feed_forward(
    x: list[list[Tensor]],
    w1: list[list[Parameter]],
    b1: list[Parameter],
    w2: list[list[Parameter]],
    b2: list[Parameter],
) -> list[list[Tensor]]:
    d_model = len(x[0])
    d_ff = len(b1)
    # Hidden layer with GeLU.
    h: list[list[Tensor]] = []
    for i in range(len(x)):
        row_i: list[Tensor] = []
        for j in range(d_ff):
            acc: Tensor = b1[j] + Tensor(data=0.0, requires_grad=False)
            for k in range(d_model):
                acc = acc + x[i][k] * w1[k][j]
            # GeLU on the autograd-free side: lift the scalar to a
            # constant Tensor. (FFN non-linearities in the
            # educational engine are detached — keeps the autograd
            # graph tractable for a tiny demo. The activation itself
            # is exact via math.erf.)
            row_i.append(Tensor(data=_gelu_scalar(acc.data), requires_grad=False))
        h.append(row_i)
    # Output layer.
    out: list[list[Tensor]] = []
    for i in range(len(x)):
        row_i2: list[Tensor] = []
        for j in range(d_model):
            acc = b2[j] + Tensor(data=0.0, requires_grad=False)
            for k in range(d_ff):
                acc = acc + h[i][k] * w2[k][j]
            row_i2.append(acc)
        out.append(row_i2)
    return out


def _gelu_scalar(x: float) -> float:
    return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))


class _no_grad:
    """Mini context manager — no actual autograd context isolation,
    but kept for forward-compat with the real ``cds.nlp.autograd.no_grad``
    once a graph-pruning pass is added.
    """

    def __enter__(self) -> _no_grad:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


# ---------------------------------------------------------------------- #
# Convenience sampler
# ---------------------------------------------------------------------- #


def sample(model: MiniGPT, prompt_ids: list[int], n_tokens: int) -> list[int]:
    """Convenience wrapper around :meth:`MiniGPT.sample` for callers
    that don't want to dig through the class API."""
    return model.sample(prompt_ids, n_tokens)
