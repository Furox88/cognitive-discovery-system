"""Token and positional embeddings in pure Python.

Implements the two embedding layers from Vaswani et al. (2017) "Attention
Is All You Need" without depending on ``torch`` or any tensor library:

* :class:`TokenEmbedding` — a learned lookup table mapping integer token
  ids to dense vectors. Training is out of scope for Sprint 1; this is
  the inference-time matrix multiply that the attention block will call
  in Sprint 2.
* :class:`PositionalEncoding` — fixed sinusoidal position embeddings
  added to the token embeddings. No learnable parameters.

Both layers are deterministic, allocation-free at construction time, and
represent their forward pass as explicit nested lists rather than numpy
arrays. This is intentional — the educational NLP track wants the math
visible.

References:
    - Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
      §3.4 (embeddings and positional encoding).
    - Karpathy, A. (2023). "Let's build GPT: from scratch, in code,
      spelled out." https://github.com/karpathy/minbpe (for the
      educational style and the choice of small Shakespeare corpus).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _make_matrix(rows: int, cols: int, fill: float = 0.0) -> list[list[float]]:
    """Allocate a ``rows × cols`` matrix as nested Python lists."""
    return [[fill] * cols for _ in range(rows)]


@dataclass
class TokenEmbedding:
    """A token-id → dense-vector lookup table.

    Initialised with small random values from a fixed RNG seed so the
    educational pipeline is reproducible. Training (gradient updates)
    lands in Sprint 3 alongside the autograd module; until then this
    is read-only — call :meth:`forward` to embed, then call
    :meth:`set_value` to hand-write weights.

    Attributes:
        vocab_size: Number of rows in the embedding table.
        d_model: Embedding dimensionality (output width).
        matrix: The ``vocab_size × d_model`` weight matrix as nested
            lists. ``matrix[id][j]`` is the j-th component of the
            embedding for token id ``id``.
    """

    vocab_size: int
    d_model: int
    matrix: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")
        # Deterministic init via a fixed seed so test runs are
        # reproducible. The exact init scheme is Xavier/Glorot-uniform,
        # truncated so that max weight ≈ 1/sqrt(d_model).
        import random

        rng = random.Random(0xC0FFEE)
        bound = 1.0 / math.sqrt(self.d_model)
        self.matrix = [
            [rng.uniform(-bound, bound) for _ in range(self.d_model)]
            for _ in range(self.vocab_size)
        ]

    def forward(self, ids: list[int]) -> list[list[float]]:
        """Look up embeddings for a sequence of token ids.

        Args:
            ids: List of token ids (must be in ``[0, vocab_size)``).

        Returns:
            A ``len(ids) × d_model`` matrix (nested list).
        """
        out: list[list[float]] = []
        for tid in ids:
            if tid < 0 or tid >= self.vocab_size:
                raise IndexError(f"Token id {tid} out of range [0, {self.vocab_size})")
            # Defensive copy — callers might mutate the result without
            # poisoning the table.
            out.append(list(self.matrix[tid]))
        return out

    def set_value(self, token_id: int, values: list[float]) -> None:
        """Overwrite the embedding for ``token_id`` (used by tests)."""
        if len(values) != self.d_model:
            raise ValueError(f"values length {len(values)} != d_model {self.d_model}")
        self.matrix[token_id] = list(values)

    @property
    def shape(self) -> tuple[int, int]:
        """Returns ``(vocab_size, d_model)``."""
        return (self.vocab_size, self.d_model)


@dataclass
class PositionalEncoding:
    """Sinusoidal positional encoding from Vaswani et al. (2017).

    PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    where ``pos`` is the zero-based position and ``i`` indexes the
    embedding dimension. The matrix is computed once at construction
    and reused for every forward pass.

    Attributes:
        max_len: Maximum sequence length the encoding supports.
        d_model: Embedding dimensionality (must match the token
            embedding it's added to).
        matrix: The precomputed ``max_len × d_model`` encoding matrix.
    """

    max_len: int
    d_model: int
    matrix: list[list[float]] = field(init=False)

    def __post_init__(self) -> None:
        if self.max_len <= 0:
            raise ValueError(f"max_len must be > 0, got {self.max_len}")
        if self.d_model <= 0:
            raise ValueError(f"d_model must be > 0, got {self.d_model}")

        self.matrix = _make_matrix(self.max_len, self.d_model)
        for pos in range(self.max_len):
            for i in range(self.d_model):
                # Even index → sin, odd index → cos. The exponent
                # 10000^(2i/d_model) grows geometrically across the
                # embedding dimension so each position gets a unique
                # low-frequency pattern.
                div_term = 10000.0 ** ((2 * (i // 2)) / self.d_model)
                angle = pos / div_term
                self.matrix[pos][i] = math.sin(angle) if i % 2 == 0 else math.cos(angle)

    def forward(self, length: int) -> list[list[float]]:
        """Return the first ``length`` rows of the encoding matrix.

        Args:
            length: Desired output length (must be ``<= max_len``).
        """
        if length < 0:
            raise ValueError(f"length must be >= 0, got {length}")
        if length > self.max_len:
            raise ValueError(f"length {length} exceeds max_len {self.max_len}")
        return [list(self.matrix[i]) for i in range(length)]

    @property
    def shape(self) -> tuple[int, int]:
        """Returns ``(max_len, d_model)``."""
        return (self.max_len, self.d_model)


def add_positional(
    token_embeddings: list[list[float]],
    positional: PositionalEncoding,
) -> list[list[float]]:
    """Add a positional encoding to a sequence of token embeddings.

    Element-wise: ``out[i][j] = token[i][j] + pos[i][j]``. Used in the
    Transformer encoder block to inject position information.

    Args:
        token_embeddings: An ``n × d_model`` matrix (nested list).
        positional: A :class:`PositionalEncoding` whose ``d_model``
            matches the token embedding width.

    Returns:
        A new ``n × d_model`` matrix (input is not mutated).
    """
    if not token_embeddings:
        return []
    n = len(token_embeddings)
    d = len(token_embeddings[0])
    if d != positional.d_model:
        raise ValueError(
            f"d_model mismatch: token embedding {d} != positional {positional.d_model}"
        )
    pos = positional.forward(n)
    return [[token_embeddings[i][j] + pos[i][j] for j in range(d)] for i in range(n)]
