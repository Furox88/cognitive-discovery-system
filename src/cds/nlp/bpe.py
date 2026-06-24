"""Byte-Pair Encoding (BPE) tokenizer in pure Python.

Implements the Sennrich et al. (2016) subword tokenisation algorithm
from scratch, without depending on ``tokenizers`` or ``sentencepiece``.
The vocabulary is built bottom-up by repeatedly merging the most
frequent adjacent symbol pair in the training corpus.

This implementation is deliberately simple and slow by design — it is
meant to be read, not benchmarked. The training loop is O(N · V · k)
where N is corpus length, V is the number of merges requested, and k is
the average pair frequency. A 1 MB corpus trains in a few seconds; a
1 GB corpus will not. Use HuggingFace ``tokenizers`` for the latter.

Design choices:

* Symbols are stored as ``tuple[str, ...]`` — one tuple per word. This
  keeps merges *non-destructive* (the original word is always
  recoverable from the encoded id sequence + vocab).
* Pair statistics use a single :class:`collections.Counter` pass per
  merge. A heap-based variant (Sennrich's original) is faster but adds
  significant code; the simple version is more honest about what's
  happening.
* The ``<unk>`` token is mandatory and always id 0. End-of-word marker
  ``</w>`` is appended to every symbol during training so the same
  character sequence can appear in different word positions without
  collapsing into the same token (e.g. ``"a"`` in ``"a cat"`` vs.
  ``"a"`` in ``"a"``).
* Encoding is greedy: scan the input, find the longest contiguous
  substring that matches any vocab entry, emit its id, advance. This
  is **not** the Sennrich-style longest-match-with-priority algorithm
  — we use plain greedy for clarity.

References:
    - Sennrich, R., Haddow, B., & Birch, A. (2016). "Neural Machine
      Translation of Rare Words with Subword Units." ACL.
    - Gage, P. (1994). "A New Algorithm for Data Compression." C Users
      Journal.
    - HuggingFace tokenizers docs — BPE section (algorithm comparison).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Reserved tokens. ``<unk>`` is mandatory (always id 0); the others are
# kept for forward-compatibility with the attention block but are not
# emitted by encode() unless ``add_specials=True``.
UNK: Final[str] = "<unk>"
PAD: Final[str] = "<pad>"
BOS: Final[str] = "<bos>"
EOS: Final[str] = "<eos>"
SPECIAL_TOKENS: Final[tuple[str, ...]] = (UNK, PAD, BOS, EOS)

# End-of-word marker. Appended to every symbol during training so the
# character "a" in " a" can differ from the "a" in "cat".
_END_OF_WORD: Final[str] = "</w>"

# Whitespace pre-tokenisation. We split on word boundaries using the
# same regex as GPT-2's basic tokenizer (simplified — contractions are
# kept together instead of split). This is good enough for educational
# use; production tokenizers use sentencepiece with byte-level fallback.
_WORD_RE: Final[re.Pattern[str]] = re.compile(r"\w+|[^\w\s]")


def _pre_tokenize(text: str) -> list[str]:
    """Split raw text into word / punctuation tokens.

    >>> _pre_tokenize("Hello, world!")
    ['Hello', ',', 'world', '!']
    """
    return _WORD_RE.findall(text)


def _word_to_symbols(word: str) -> tuple[str, ...]:
    """Convert a word into its initial character symbols.

    >>> _word_to_symbols("cat")
    ('c', 'a', 't', '</w>')
    """
    return tuple(list(word) + [_END_OF_WORD])


def _get_pair_stats(
    corpus: list[tuple[str, ...]],
) -> Counter[tuple[str, str]]:
    """Count adjacent symbol-pair frequencies across the corpus."""
    stats: Counter[tuple[str, str]] = Counter()
    for symbols in corpus:
        for a, b in zip(symbols, symbols[1:]):
            stats[(a, b)] += 1
    return stats


def _merge_pair(
    corpus: list[tuple[str, ...]],
    pair: tuple[str, str],
) -> list[tuple[str, ...]]:
    """Replace every occurrence of ``pair`` with its concatenation."""
    a, b = pair
    merged = a + b
    new_corpus: list[tuple[str, ...]] = []
    for symbols in corpus:
        if len(symbols) < 2:
            new_corpus.append(symbols)
            continue
        out: list[str] = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_corpus.append(tuple(out))
    return new_corpus


@dataclass
class BPEMerge:
    """A single BPE merge rule.

    Attributes:
        pair: The adjacent symbol pair that was merged.
        rank: Priority — lower rank = applied earlier. When two merges
            could both apply, the lower-rank one wins. (Greedy encoding
            relies on this.)
        new_token: The merged token string (= ``pair[0] + pair[1]``).
    """

    pair: tuple[str, str]
    rank: int
    new_token: str

    def to_dict(self) -> dict[str, object]:
        """Serialize this merge to a JSON-friendly dict."""
        return {"pair": list(self.pair), "rank": self.rank, "new_token": self.new_token}

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BPEMerge:
        """Reconstruct a :class:`BPEMerge` from :meth:`to_dict` output.

        Raises:
            ValueError: if ``data`` is missing keys, has the wrong types,
                or the ``pair`` does not contain exactly two strings.
        """
        pair_raw = data["pair"]
        if not isinstance(pair_raw, list) or len(pair_raw) != 2:
            raise ValueError(f"Invalid pair in BPE merge: {pair_raw!r}")
        a_raw, b_raw = pair_raw[0], pair_raw[1]
        if not isinstance(a_raw, str) or not isinstance(b_raw, str):
            raise ValueError(f"Invalid pair components: {pair_raw!r}")
        rank_raw = data["rank"]
        if not isinstance(rank_raw, int) or isinstance(rank_raw, bool):
            raise ValueError(f"Invalid rank: {rank_raw!r}")
        token_raw = data["new_token"]
        if not isinstance(token_raw, str):
            raise ValueError(f"Invalid new_token: {token_raw!r}")
        return cls(pair=(a_raw, b_raw), rank=rank_raw, new_token=token_raw)


@dataclass
class BPETokenizer:
    """A trained byte-pair encoding tokenizer.

    Construct one with :func:`train_bpe`, or load a previously saved one
    with :meth:`load`. Encoding is greedy: scan input left-to-right,
    find the longest contiguous substring that is in the vocab, emit
    its id, advance. Unknown characters fall back to ``<unk>`` (id 0).

    Attributes:
        vocab: Mapping from token string → integer id.
        id_to_token: Inverse of ``vocab`` (built lazily on demand).
        merges: List of BPE merge rules, in training order. The index
            of a rule is its priority (lower = earlier).
        eow: End-of-word marker symbol appended during training.
    """

    vocab: dict[str, int] = field(default_factory=dict)
    merges: list[BPEMerge] = field(default_factory=list)
    eow: str = _END_OF_WORD

    def __post_init__(self) -> None:
        """Build and cache the inverse (id → token) vocabulary for fast decoding."""
        # Build the inverse vocabulary. We keep it cached so decode() is
        # O(N) rather than O(N log V).
        self._id_to_token: dict[int, str] = {i: t for t, i in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary (including specials)."""
        return len(self.vocab)

    @property
    def id_to_token(self) -> dict[int, str]:
        """Read-only view of id → token mapping."""
        return dict(self._id_to_token)

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #

    def encode(self, text: str, add_specials: bool = False) -> list[int]:
        """Encode ``text`` into a list of token ids.

        Greedy longest-match: scan the input, at each position take the
        longest substring that exists in the vocabulary. If no character
        matches, emit ``<unk>`` and advance by one.

        Args:
            text: Raw input string. Will be pre-tokenised on whitespace
                and punctuation.
            add_specials: If True, prepend ``<bos>`` (id 2) and append
                ``<eos>`` (id 3). Off by default — the educational
                pipeline wants raw token streams.
        """
        if not self.vocab:
            raise ValueError("Vocabulary is empty. Train or load a tokenizer first.")

        words = _pre_tokenize(text)
        ids: list[int] = []
        if add_specials:
            bos_id = self.vocab.get(BOS)
            if bos_id is not None:
                ids.append(bos_id)

        for word in words:
            ids.extend(self._encode_word(word))

        if add_specials:
            eos_id = self.vocab.get(EOS)
            if eos_id is not None:
                ids.append(eos_id)
        return ids

    def _encode_word(self, word: str) -> list[int]:
        """Greedy-encode a single pre-tokenised word.

        We try every contiguous substring starting at the current cursor,
        longest first, and pick the first one that's in the vocab. This
        is O(L²) per word but L is small (typical English word ≤ 20 chars).
        """
        symbols = _word_to_symbols(word)
        result: list[int] = []
        i = 0
        n = len(symbols)
        while i < n:
            # Longest contiguous substring starting at i that is in vocab.
            matched = False
            # Walk from i+1 to i+n inclusive, looking for the longest
            # substring that exists in vocab.
            for j in range(n, i, -1):
                candidate = "".join(symbols[i:j])
                token_id = self.vocab.get(candidate)
                if token_id is not None:
                    result.append(token_id)
                    i = j
                    matched = True
                    break
            if not matched:
                # Single character not in vocab — emit unk and skip it.
                # This is unreachable in practice because every training
                # character is in the base vocab, but keep the guard.
                unk_id = self.vocab.get(UNK, 0)
                result.append(unk_id)
                i += 1
        return result

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #

    def decode(self, ids: list[int], strip_eow: bool = True) -> str:
        """Decode a list of ids back to a string.

        Args:
            ids: Token id sequence (must be non-negative integers in
                ``[0, vocab_size)``).
            strip_eow: If True (default), drop the ``</w>`` marker from
                the end of each word and join with spaces. Works for
                both the literal ``</w>`` token *and* merged tokens
                that end in ``</w>`` (e.g. ``"low</w>"``). If False,
                concatenate the raw token strings.
        """
        if strip_eow:
            words: list[str] = []
            current: list[str] = []
            for tid in ids:
                tok = self._id_to_token.get(tid)
                if tok is None:
                    raise ValueError(f"Unknown token id: {tid}")
                if tok == UNK:
                    current.append("�")
                elif tok.endswith(_END_OF_WORD):
                    # Either a literal eow token ("</w>") or a merged
                    # token that ends with it ("low</w>"). Flush the
                    # current word minus the suffix.
                    stem = tok[: -len(_END_OF_WORD)]
                    current.append(stem)
                    words.append("".join(current))
                    current = []
                else:
                    current.append(tok)
            if current:
                words.append("".join(current))
            # Collapse empty strings (from consecutive EOS/PAD tokens)
            words = [w for w in words if w]
            return " ".join(words)
        # Raw concatenation — useful for inspecting token boundaries.
        return "".join(self._id_to_token.get(tid, "") for tid in ids)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str | Path) -> None:
        """Save the tokenizer to a JSON file.

        Format::

            {
                "vocab": {"<unk>": 0, "a": 1, ...},
                "merges": [{"pair": ["a", "b"], "rank": 0, "new_token": "ab"}, ...],
                "eow": "</w>"
            }
        """
        payload = {
            "vocab": self.vocab,
            "merges": [m.to_dict() for m in self.merges],
            "eow": self.eow,
        }
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer previously saved with :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges_raw = data.get("merges", [])
        if not isinstance(merges_raw, list):
            raise ValueError("Invalid merges field in tokenizer file")
        merges: list[BPEMerge] = []
        for raw_merge in merges_raw:
            if not isinstance(raw_merge, dict):
                raise ValueError(f"Invalid merge entry: {raw_merge!r}")
            merges.append(BPEMerge.from_dict(raw_merge))
        return cls(
            vocab=dict(data["vocab"]),
            merges=merges,
            eow=str(data.get("eow", _END_OF_WORD)),
        )


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #


def train_bpe(
    corpus: str,
    vocab_size: int = 1000,
    min_frequency: int = 2,
    show_progress: bool = False,
) -> BPETokenizer:
    """Train a BPE tokenizer on a raw text corpus.

    The training procedure follows Sennrich et al. (2016) — start with
    a base vocabulary of every distinct character in the corpus, then
    repeatedly merge the most frequent adjacent symbol pair until the
    vocabulary reaches ``vocab_size``.

    Args:
        corpus: Raw training text. UTF-8 string; will be normalised to
            NFC implicitly by Python's str handling.
        vocab_size: Target vocabulary size (including the four reserved
            special tokens). The actual vocabulary may be smaller if
            the corpus has fewer unique characters — training stops when
            no pair exceeds ``min_frequency``.
        min_frequency: Stop merging when the most frequent pair has
            count ≤ this. Prevents pathological merges from rare noise.
        show_progress: If True, print each merge as it happens. Off by
            default for clean test output.

    Returns:
        A fully populated :class:`BPETokenizer`.

    Raises:
        ValueError: If ``corpus`` is empty or ``vocab_size < 5`` (must
            fit the four reserved tokens + at least one real token).

    Example:
        >>> tk = train_bpe("low low low lower lower newest newest newest", vocab_size=20)
        >>> "low" in tk.vocab and "est" in tk.vocab
        True
    """
    if not corpus:
        raise ValueError("Cannot train BPE on an empty corpus")
    if vocab_size < 5:
        raise ValueError(f"vocab_size must be >= 5 (4 specials + 1 real), got {vocab_size}")

    # 1. Base vocabulary: every distinct character in the corpus.
    base_chars = sorted(set(corpus))
    vocab: dict[str, int] = {tok: idx for idx, tok in enumerate(SPECIAL_TOKENS)}
    next_id = len(SPECIAL_TOKENS)
    for ch in base_chars:
        # SPECIAL_TOKENS are all multi-char strings, so a single ``ch`` can
        # never collide with them — the False branch is unreachable. Kept as a
        # defensive guard against a future single-char special token.
        if ch not in vocab:  # pragma: no branch
            vocab[ch] = next_id
            next_id += 1

    # 2. Tokenise corpus into per-word symbol sequences.
    words = _pre_tokenize(corpus)
    if not words:
        # Corpus contains only whitespace / punctuation.
        return BPETokenizer(vocab=vocab, merges=[])

    corpus_symbols: list[tuple[str, ...]] = [_word_to_symbols(w) for w in words]

    # 3. Iteratively merge the most frequent pair.
    merges: list[BPEMerge] = []
    while len(vocab) < vocab_size:
        stats = _get_pair_stats(corpus_symbols)
        if not stats:
            break
        best_pair, best_count = stats.most_common(1)[0]
        if best_count < min_frequency:
            break

        new_token = best_pair[0] + best_pair[1]
        # A merged token can never already be in vocab: stats only counts pairs
        # of currently-unmerged symbols, so a previously-merged token is not a
        # candidate. The False branch is unreachable; kept defensively.
        if new_token not in vocab:  # pragma: no branch
            vocab[new_token] = next_id
            next_id += 1
        merges.append(BPEMerge(pair=best_pair, rank=len(merges), new_token=new_token))
        if show_progress:
            print(f"merge {len(merges):>4}: {best_pair!r} -> {new_token!r} (count={best_count})")

        corpus_symbols = _merge_pair(corpus_symbols, best_pair)

    return BPETokenizer(vocab=vocab, merges=merges)
