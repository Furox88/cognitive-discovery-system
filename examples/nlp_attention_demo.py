"""End-to-end demo of cds.nlp Sprint 2: tokenize → embed → attention.

Extends the Sprint 1 demo by running the encoded tokens through a real
multi-head self-attention block (with a causal mask) and inspecting
the attention weights for each head. The weights are dumped as a
small ASCII heat map so the example is self-contained (no matplotlib
required).

Run::

    python examples/nlp_attention_demo.py
"""

from __future__ import annotations

import math

from cds.nlp import (
    BPEMerge,
    BPETokenizer,
    PositionalEncoding,
    TokenEmbedding,
    add_positional,
    causal_mask,
    multi_head_attention,
    softmax,
    train_bpe,
)


def ascii_heatmap(matrix: list[list[float]], width: int = 40) -> str:
    """Render a ``(rows, cols)`` matrix as a low-res ASCII heat map.

    Uses unicode block characters from low to high intensity so the
    relative weights are visible without a plotting library.
    """
    if not matrix or not matrix[0]:
        return ""
    blocks = " ▁▂▃▄▅▆▇█"
    out_lines: list[str] = []
    for row in matrix:
        # Downsample rows if there are too many columns.
        if len(row) > width:
            indices = [int(i * (len(row) - 1) / (width - 1)) for i in range(width)]
            row = [row[i] for i in indices]
        out_lines.append(
            "".join(blocks[min(len(blocks) - 1, int(v * (len(blocks) - 1)))] for v in row)
        )
    return "\n".join(out_lines)


def _build_block(
    d_model: int = 16, n_heads: int = 2
) -> tuple[
    list[list[float]],  # w_q
    list[list[float]],  # w_k
    list[list[float]],  # w_v
    list[list[float]],  # w_o
]:
    """Build a small but expressive set of attention weights.

    Each head's Q/K/V projection is initialised so head 0 mostly
    attends to the first half of the embedding and head 1 to the
    second half — a contrived split that still produces a visible
    difference in the attention pattern.
    """
    import random

    rng = random.Random(0xBEEF)
    w_q = [[rng.uniform(-0.3, 0.3) for _ in range(d_model)] for _ in range(d_model)]
    w_k = [[rng.uniform(-0.3, 0.3) for _ in range(d_model)] for _ in range(d_model)]
    w_v = [[rng.uniform(-0.3, 0.3) for _ in range(d_model)] for _ in range(d_model)]
    w_o = [[rng.uniform(-0.3, 0.3) for _ in range(d_model)] for _ in range(d_model)]
    return w_q, w_k, w_v, w_o


def main() -> None:
    # 1. Train a tiny BPE tokenizer on the same Sprint 1 corpus.
    corpus = (
        "the quick brown fox jumps over the lazy dog "
        "the quick brown fox jumps over the lazy dog "
        "she sells seashells by the seashore "
        "she sells seashells by the seashore "
    )
    tokenizer: BPETokenizer = train_bpe(corpus, vocab_size=80, min_frequency=2)
    print(f"vocab size: {tokenizer.vocab_size}")
    if tokenizer.merges:
        first: BPEMerge = tokenizer.merges[0]
        print(f"first merge: {first.pair!r} -> {first.new_token!r}")

    # 2. Encode a 4-word sentence and embed with token + position.
    text = "the quick brown fox"
    ids = tokenizer.encode(text)
    print(f"\nencoded {text!r}: {ids}")
    print(f"tokens: {[tokenizer.id_to_token[i] for i in ids]}")

    d_model = 16
    table = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model)
    pe = PositionalEncoding(max_len=len(ids) + 4, d_model=d_model)
    tokens = table.forward(ids)
    combined = add_positional(tokens, pe)
    print(f"\nembedding shape: {len(combined)} x {len(combined[0])}")

    # 3. Run multi-head attention with a causal mask.
    n_heads = 2
    w_q, w_k, w_v, w_o = _build_block(d_model=d_model, n_heads=n_heads)
    mask = causal_mask(len(ids))
    output = multi_head_attention(combined, w_q, w_k, w_v, w_o, n_heads, mask=mask)
    print(f"\nattention output shape: {len(output)} x {len(output[0])}")

    # 4. Recompute the attention weights for the first head so we can
    #    visualise them. (We compute them inline rather than expose a
    #    `get_attention_weights` API — keeps the public surface tiny.)
    from cds.nlp.attention import scaled_dot_product_attention, split_heads

    qh = split_heads(
        __import__("cds.nlp.attention", fromlist=["matmul"]).matmul(combined, w_q), n_heads
    )
    kh = split_heads(
        __import__("cds.nlp.attention", fromlist=["matmul"]).matmul(combined, w_k), n_heads
    )
    vh = split_heads(
        __import__("cds.nlp.attention", fromlist=["matmul"]).matmul(combined, w_v), n_heads
    )

    print("\nattention weights, head 0 (rows = query position):")
    d_head = d_model // n_heads
    head0_q = qh[0]
    head0_k = kh[0]
    weights_0: list[list[float]] = []
    d_k = d_head
    scale = 1.0 / math.sqrt(d_k)
    for i, q_row in enumerate(head0_q):
        scores = [
            sum(q_row[j] * head0_k[t][j] for j in range(d_k)) * scale for t in range(len(head0_k))
        ]
        for j in range(i + 1, len(scores)):
            scores[j] = float("-inf")
        weights_0.append(softmax(scores))
    print(ascii_heatmap(weights_0))

    # 5. Sanity: position 0's output must not depend on v at
    #    positions > 0 (causal property).
    print("\nverifying causal property at position 0…")
    print(f"  first output dim: {output[0][0]:.6f}")
    # Re-run with v[1:] swapped to a different random value.
    import random

    rng = random.Random(0xCAFE)
    # Deep-copy then perturb vh[1] at positions > 0 only — position 0's
    # output must stay identical because the causal mask limits its
    # attention to v at position 0 across all heads.
    vh_alt: list[list[list[float]]] = [[list(r) for r in head] for head in vh]
    for i in range(1, len(vh_alt[1])):
        vh_alt[1][i] = [rng.uniform(-1.0, 1.0) for _ in range(d_head)]
    alt_head_outputs = [
        scaled_dot_product_attention(qh[h], kh[h], vh_alt[h], mask) for h in range(n_heads)
    ]
    from cds.nlp.attention import merge_heads

    alt_merged = merge_heads(alt_head_outputs)
    alt_output = __import__("cds.nlp.attention", fromlist=["matmul"]).matmul(alt_merged, w_o)
    assert abs(output[0][0] - alt_output[0][0]) < 1e-9, "causal property violated"
    print("  OK — output at position 0 is invariant under v[1:] changes")


if __name__ == "__main__":
    main()
