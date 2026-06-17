"""Natural language processing primitives in pure Python.

Educational, from-scratch implementations of the building blocks used in
modern language models — byte-pair encoding (BPE) tokenisation and the
sinusoidal token / positional embeddings from the original Transformer
paper. Designed for teaching, prototyping, and small-model experiments,
not for production-scale training (no NumPy, no BLAS — performance is
deliberately the trade-off for full transparency).

Why this module exists inside CDS:
- Demonstrates that the *core ideas* of transformer-era NLP are short,
  readable pure-Python code (BPE ≈ 250 lines, embeddings ≈ 150 lines).
- Provides a reproducible playground for tokenisation experiments
  without pulling in tokenizers / sentencepiece / torch.
- Pairs naturally with :mod:`cds.optimization` (loss + LR schedule) and
  :mod:`cds.probability` (softmax, sampling) when the attention block
  lands in Sprint 2.

Scope (Sprint 1 — v0.9.0b5):

* :func:`~cds.nlp.bpe.train_bpe` — train a BPE vocabulary from a corpus
* :class:`~cds.nlp.bpe.BPETokenizer` — encode / decode / save / load
* :class:`~cds.nlp.embed.TokenEmbedding` — token lookup table
* :class:`~cds.nlp.embed.PositionalEncoding` — sinusoidal positions

Out of scope for the educational track:
- Backpropagation / autograd (lives in Sprint 3, with Numba as optional
  JIT acceleration behind ``cds[fast-jit]``; the core stays pure Python).
- Subword sampling tricks (BPE-Dropout, Unigram LM).
- WordPiece / SentencePiece alternatives.

References:
    - Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine
      Translation of Rare Words with Subword Units." Proceedings of the
      54th Annual Meeting of the ACL.
    - Gage, P. (1994). "A New Algorithm for Data Compression." C Users
      Journal, 12(2). (Original BPE description.)
    - Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
      §3.4 (positional encoding).
"""

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
]
