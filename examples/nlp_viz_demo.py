"""End-to-end demo of ``cds.nlp`` visualisation primitives.

Trains a tiny BPE tokenizer, runs a short attention + embedding forward
pass, then renders all three ASCII visualisations. Runs in under 2s with
no optional dependencies.

Run::

    python examples/nlp_viz_demo.py
"""

from __future__ import annotations

import math

from cds.nlp import (
    PositionalEncoding,
    TokenEmbedding,
    add_positional,
    render_attention_heatmap,
    render_embedding_projection,
    render_training_curve,
    softmax,
    train_bpe,
)

# ``cds.nlp.matmul`` is the autograd (Tensor) matmul; the float-matrix matmul
# lives in the attention module and is what we need for the score product.
from cds.nlp.attention import matmul as attn_matmul
from cds.nlp.attention import transpose


def main() -> None:
    # 1. Tokenise a short sentence.
    corpus = "the quick brown fox the lazy dog the quick brown fox "
    tokenizer = train_bpe(corpus, vocab_size=40, min_frequency=1)
    text = "the quick brown fox"
    ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_token[i] for i in ids]
    print(f"text:   {text!r}")
    print(f"tokens: {tokens}")

    # 2. Embed (d_model=8) + add positional encoding.
    #    TokenEmbedding / PositionalEncoding are dataclasses whose public
    #    forward pass is the ``.forward`` method (no ``__call__``), and
    #    ``add_positional`` takes the PositionalEncoding object directly.
    d_model = 8
    embed = TokenEmbedding(vocab_size=tokenizer.vocab_size, d_model=d_model)
    pos = PositionalEncoding(max_len=len(ids), d_model=d_model)
    x = add_positional(embed.forward(ids), pos)

    # 3. Attention weights via scaled dot-product scores (single head).
    #    ``scaled_dot_product_attention`` returns the (weights @ V) output, not
    #    the weight matrix, so we build the weights explicitly: softmax of the
    #    scaled Q K^T scores. This mirrors what the renderer consumes.
    d_k = len(x[0])
    scale = 1.0 / math.sqrt(d_k)
    scores = attn_matmul(x, transpose(x))
    attn = [softmax([s * scale for s in row]) for row in scores]
    print("\n--- Attention heatmap ---")
    print(render_attention_heatmap(attn, tokens, tokens))

    # 4. Embedding projection: show part of the vocab in 2-D.
    limit = min(12, tokenizer.vocab_size)
    vocab_vectors = [embed.forward([i])[0] for i in range(limit)]
    vocab_labels = [tokenizer.id_to_token[i] or f"<{i}>" for i in range(limit)]
    print("--- Embedding projection (top-12 vocab) ---")
    print(render_embedding_projection(vocab_vectors, labels=vocab_labels, top_n=12))

    # 5. A pretend training curve.
    losses = [
        3.50,
        3.10,
        2.80,
        2.55,
        2.30,
        2.10,
        1.95,
        1.82,
        1.70,
        1.60,
        1.52,
        1.45,
        1.39,
        1.34,
        1.30,
        1.27,
        1.24,
        1.22,
        1.20,
        1.19,
    ]
    print("--- Training loss curve ---")
    print(render_training_curve(losses, width=50, height=10))


if __name__ == "__main__":
    main()
