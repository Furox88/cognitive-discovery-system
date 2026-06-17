"""Natural language processing primitives in pure Python.

Educational, from-scratch implementations of the building blocks used in
modern language models — byte-pair encoding (BPE) tokenisation, the
sinusoidal token / positional embeddings from the original Transformer
paper, scaled dot-product and multi-head self-attention, and the
Transformer encoder block (GeLU FFN, LayerNorm, residual).

Designed for teaching, prototyping, and small-model experiments, not
for production-scale training (no NumPy, no BLAS — performance is
deliberately the trade-off for full transparency).

Why this module exists inside CDS:
- Demonstrates that the *core ideas* of transformer-era NLP are short,
  readable pure-Python code.
- Provides a reproducible playground for tokenisation experiments
  without pulling in tokenizers / sentencepiece / torch.
- Pairs naturally with :mod:`cds.optimization` (loss + LR schedule)
  and :mod:`cds.probability` (softmax, sampling) when the autograd
  module lands in Sprint 3.

Scope (Sprint 1+2 — v0.9.0b8):

* :func:`~cds.nlp.bpe.train_bpe` — train a BPE vocabulary from a corpus
* :class:`~cds.nlp.bpe.BPETokenizer` — encode / decode / save / load
* :class:`~cds.nlp.embed.TokenEmbedding` — token lookup table
* :class:`~cds.nlp.embed.PositionalEncoding` — sinusoidal positions
* :func:`~cds.nlp.attention.scaled_dot_product_attention`
* :func:`~cds.nlp.attention.multi_head_attention`
* :func:`~cds.nlp.attention.causal_mask` — decoder self-attention mask
* :func:`~cds.nlp.layers.gelu` / :func:`~cds.nlp.layers.layer_norm`
* :func:`~cds.nlp.layers.feed_forward`
* :func:`~cds.nlp.layers.transformer_block`

Out of scope for the educational track:
- Backpropagation / autograd (lives in Sprint 3, with Numba as
  optional JIT acceleration behind ``cds[fast-jit]``; the core
  stays pure Python).
- Dropout, ALiBi, RoPE, GQA / MQA — modern attention refinements
  tracked for a later educational add-on.
- Subword sampling tricks (BPE-Dropout, Unigram LM).
- WordPiece / SentencePiece alternatives.

References:
    - Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine
      Translation of Rare Words with Subword Units." ACL.
    - Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
    - Gage, P. (1994). "A New Algorithm for Data Compression." C
      Users Journal.
"""

from cds.nlp.attention import (
    causal_mask,
    matmul,
    merge_heads,
    multi_head_attention,
    scaled_dot_product_attention,
    softmax,
    split_heads,
    transpose,
)
from cds.nlp.bpe import (
    BOS,
    EOS,
    PAD,
    SPECIAL_TOKENS,
    UNK,
    BPEMerge,
    BPETokenizer,
    train_bpe,
)
from cds.nlp.embed import (
    PositionalEncoding,
    TokenEmbedding,
    add_positional,
)
from cds.nlp.layers import (
    feed_forward,
    gelu,
    layer_norm,
    transformer_block,
)

__all__ = [
    # bpe
    "UNK",
    "PAD",
    "BOS",
    "EOS",
    "SPECIAL_TOKENS",
    "BPEMerge",
    "BPETokenizer",
    "train_bpe",
    # embed
    "TokenEmbedding",
    "PositionalEncoding",
    "add_positional",
    # attention
    "softmax",
    "matmul",
    "transpose",
    "scaled_dot_product_attention",
    "split_heads",
    "merge_heads",
    "multi_head_attention",
    "causal_mask",
    # layers
    "gelu",
    "layer_norm",
    "feed_forward",
    "transformer_block",
]
