"""Tests for :mod:`cds.nlp.bpe`.

Covers training, encoding / decoding round-trips, edge cases (empty
inputs, all-whitespace corpora, repeated characters), persistence, and
the deterministic behaviour of the merge priority order.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cds.nlp.bpe import (
    BOS,
    EOS,
    SPECIAL_TOKENS,
    UNK,
    BPEMerge,
    BPETokenizer,
    train_bpe,
)

# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture
def small_corpus() -> str:
    """A short, deterministic corpus with predictable merge statistics."""
    return "low low low lower lower newest newest newest newest widest widest"


@pytest.fixture
def small_tokenizer(small_corpus: str) -> BPETokenizer:
    """A tokenizer trained on ``small_corpus`` with a small vocabulary."""
    return train_bpe(small_corpus, vocab_size=40, min_frequency=2)


# ---------------------------------------------------------------------- #
# Training
# ---------------------------------------------------------------------- #


class TestTrainBPE:
    """The training procedure itself."""

    def test_returns_tokenizer_with_vocab(self, small_corpus: str) -> None:
        tk = train_bpe(small_corpus, vocab_size=40)
        assert isinstance(tk, BPETokenizer)
        assert tk.vocab_size > 0

    def test_vocab_contains_all_specials(self, small_corpus: str) -> None:
        tk = train_bpe(small_corpus, vocab_size=40)
        for tok in SPECIAL_TOKENS:
            assert tok in tk.vocab

    def test_unk_is_id_zero(self, small_corpus: str) -> None:
        tk = train_bpe(small_corpus, vocab_size=40)
        assert tk.vocab[UNK] == 0

    def test_vocab_ids_are_unique(self, small_corpus: str) -> None:
        tk = train_bpe(small_corpus, vocab_size=40)
        ids = list(tk.vocab.values())
        assert len(ids) == len(set(ids)), "vocab ids must be unique"

    def test_base_chars_present_in_vocab(self, small_corpus: str) -> None:
        """Every distinct character in the corpus must be in the base vocab."""
        tk = train_bpe(small_corpus, vocab_size=10_000)
        for ch in set(small_corpus):
            assert ch in tk.vocab, f"missing base char {ch!r}"

    def test_empty_corpus_raises(self) -> None:
        with pytest.raises(ValueError, match="empty corpus"):
            train_bpe("", vocab_size=10)

    def test_vocab_size_too_small_raises(self) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            train_bpe("hello", vocab_size=4)

    def test_min_frequency_stops_merges(self, small_corpus: str) -> None:
        """A high min_frequency should stop training before vocab_size."""
        low_threshold = train_bpe(small_corpus, vocab_size=200, min_frequency=1)
        high_threshold = train_bpe(small_corpus, vocab_size=200, min_frequency=10)
        assert high_threshold.vocab_size <= low_threshold.vocab_size

    def test_higher_target_vocab_yields_more_merges(self, small_corpus: str) -> None:
        small = train_bpe(small_corpus, vocab_size=20)
        larger = train_bpe(small_corpus, vocab_size=200)
        assert larger.vocab_size >= small.vocab_size
        assert len(larger.merges) >= len(small.merges)

    def test_merges_have_unique_pairs(self, small_corpus: str) -> None:
        """Each merge rule targets a distinct adjacent pair."""
        tk = train_bpe(small_corpus, vocab_size=40)
        pairs = [m.pair for m in tk.merges]
        assert len(pairs) == len(set(pairs))

    def test_merges_are_ordered(self, small_corpus: str) -> None:
        """Merge rank must match its index in the list (greedy priority)."""
        tk = train_bpe(small_corpus, vocab_size=40)
        for i, m in enumerate(tk.merges):
            assert m.rank == i

    def test_merge_new_token_equals_concat(self, small_corpus: str) -> None:
        tk = train_bpe(small_corpus, vocab_size=40)
        for m in tk.merges:
            assert m.new_token == m.pair[0] + m.pair[1]


# ---------------------------------------------------------------------- #
# Encoding
# ---------------------------------------------------------------------- #


class TestEncode:
    """``BPETokenizer.encode`` correctness."""

    def test_ids_are_non_negative(self, small_tokenizer: BPETokenizer) -> None:
        ids = small_tokenizer.encode("low")
        assert all(isinstance(i, int) and i >= 0 for i in ids)

    def test_ids_within_vocab(self, small_tokenizer: BPETokenizer) -> None:
        ids = small_tokenizer.encode("low lower newest widest")
        for tid in ids:
            assert tid < small_tokenizer.vocab_size

    def test_empty_string_returns_empty(self, small_tokenizer: BPETokenizer) -> None:
        assert small_tokenizer.encode("") == []

    def test_punctuation_handled(self, small_tokenizer: BPETokenizer) -> None:
        """Punctuation tokens (``_pre_tokenize`` regex output) encode to
        one or more ids — never silently dropped."""
        ids = small_tokenizer.encode("low!")
        assert ids, "punctuation must produce at least one id"
        assert all(i < small_tokenizer.vocab_size for i in ids)

    def test_add_specials_prepends_bos(self, small_tokenizer: BPETokenizer) -> None:
        ids_with = small_tokenizer.encode("low", add_specials=True)
        ids_without = small_tokenizer.encode("low")
        assert ids_with[0] == small_tokenizer.vocab[BOS]
        # Strip BOS prefix AND EOS suffix — the content slice must match.
        assert ids_with[1:-1] == ids_without

    def test_add_specials_appends_eos(self, small_tokenizer: BPETokenizer) -> None:
        ids = small_tokenizer.encode("low", add_specials=True)
        assert ids[-1] == small_tokenizer.vocab[EOS]

    def test_repeated_words_compress(self) -> None:
        """A larger vocabulary compresses frequent words into single tokens."""
        # vocab_size=10 leaves room only for the 4 specials + base chars.
        # vocab_size=100 lets frequent subwords (e.g. 'low</w>') merge.
        tiny = train_bpe("low " * 200, vocab_size=10)
        big = train_bpe("low " * 200, vocab_size=100)
        assert len(big.encode("low")) < len(tiny.encode("low"))

    def test_unknown_chars_use_unk(self) -> None:
        """Corpus has no Chinese chars; encoding them should still produce
        valid ids via the unknown-character fallback."""
        tk = train_bpe("hello world " * 50, vocab_size=50)
        ids = tk.encode("你好")
        assert ids  # non-empty
        assert all(0 <= i < tk.vocab_size for i in ids)


# ---------------------------------------------------------------------- #
# Decoding & round-trip
# ---------------------------------------------------------------------- #


class TestDecode:
    """``BPETokenizer.decode`` and round-trip safety."""

    def test_decode_returns_string(self, small_tokenizer: BPETokenizer) -> None:
        out = small_tokenizer.decode(small_tokenizer.encode("low lower"))
        assert isinstance(out, str)

    def test_round_trip_preserves_words(self, small_tokenizer: BPETokenizer) -> None:
        """The decoded text contains all the original words (whitespace
        may shift because BPE inserts word separators)."""
        original = "low lower newest"
        decoded = small_tokenizer.decode(small_tokenizer.encode(original))
        for word in original.split():
            assert word in decoded, f"{word!r} lost in round-trip"

    def test_decode_unknown_id_raises(self, small_tokenizer: BPETokenizer) -> None:
        with pytest.raises(ValueError, match="Unknown token id"):
            small_tokenizer.decode([999_999])

    def test_decode_negative_id_raises(self, small_tokenizer: BPETokenizer) -> None:
        with pytest.raises(ValueError, match="Unknown token id"):
            small_tokenizer.decode([-1])

    def test_decode_raw_no_eow(self, small_tokenizer: BPETokenizer) -> None:
        ids = small_tokenizer.encode("low")
        raw = small_tokenizer.decode(ids, strip_eow=False)
        assert isinstance(raw, str)
        # Raw form must contain the end-of-word marker somewhere.
        assert small_tokenizer.eow in raw

    def test_decode_idempotent(self, small_tokenizer: BPETokenizer) -> None:
        """decode(encode(x)) for a single word is a stable round-trip."""
        # Single word only — multi-word decode emits </w> markers that
        # the re-encode step doesn't recognise as literals.
        first = small_tokenizer.decode(small_tokenizer.encode("low"))
        second = small_tokenizer.decode(small_tokenizer.encode(first))
        assert first == second == "low"


# ---------------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------------- #


class TestPersistence:
    """Save / load round-trip preserves the tokenizer exactly."""

    def test_save_and_load(self, small_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        path = tmp_path / "tokenizer.json"
        small_tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.vocab == small_tokenizer.vocab
        assert len(loaded.merges) == len(small_tokenizer.merges)
        assert loaded.eow == small_tokenizer.eow

    def test_save_produces_json(self, small_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        path = tmp_path / "tokenizer.json"
        small_tokenizer.save(path)
        # File must be valid JSON.
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "vocab" in data
        assert "merges" in data

    def test_load_invalid_pair_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"vocab": {"a": 1}, "merges": [{"pair": "abc", "rank": 0}]}))
        with pytest.raises(ValueError, match="pair"):
            BPETokenizer.load(path)

    def test_loaded_encode_matches(self, small_tokenizer: BPETokenizer, tmp_path: Path) -> None:
        path = tmp_path / "tokenizer.json"
        small_tokenizer.save(path)
        loaded = BPETokenizer.load(path)
        text = "low lower newest widest"
        assert loaded.encode(text) == small_tokenizer.encode(text)

    def test_unicode_in_corpus(self, tmp_path: Path) -> None:
        tk = train_bpe("café café café résumé résumé", vocab_size=60)
        path = tmp_path / "u.json"
        tk.save(path)
        loaded = BPETokenizer.load(path)
        # Round-trip must not lose the é.
        decoded = loaded.decode(loaded.encode("café résumé"))
        assert "café" in decoded
        assert "résumé" in decoded


# ---------------------------------------------------------------------- #
# Edge cases
# ---------------------------------------------------------------------- #


class TestEdgeCases:
    """Defensive paths and unusual inputs."""

    def test_whitespace_only_corpus(self) -> None:
        """A corpus with no word characters must not crash."""
        tk = train_bpe("   \n\t  ", vocab_size=10)
        # Vocab contains at least the specials + maybe space-like chars.
        assert tk.vocab_size >= len(SPECIAL_TOKENS)

    def test_single_character_corpus(self) -> None:
        # min_frequency=10 prevents the (a, a) pair (count=3) from merging.
        tk = train_bpe("aaaa", vocab_size=10, min_frequency=10)
        ids = tk.encode("aa")
        assert ids
        assert len(tk.merges) == 0

    def test_encode_uninitialised_raises(self) -> None:
        tk = BPETokenizer()
        with pytest.raises(ValueError, match="empty"):
            tk.encode("hello")

    @pytest.mark.parametrize("text", ["hello", "world", "x", "abc def"])
    def test_encode_idempotent(self, text: str, small_tokenizer: BPETokenizer) -> None:
        """Encoding twice produces the same id sequence."""
        a = small_tokenizer.encode(text)
        b = small_tokenizer.encode(text)
        assert a == b

    def test_vocab_size_property(self, small_tokenizer: BPETokenizer) -> None:
        assert small_tokenizer.vocab_size == len(small_tokenizer.vocab)
        assert small_tokenizer.vocab_size == len(small_tokenizer.id_to_token)

    def test_id_to_token_is_inverse(self, small_tokenizer: BPETokenizer) -> None:
        for tok, tid in small_tokenizer.vocab.items():
            assert small_tokenizer.id_to_token[tid] == tok


# ---------------------------------------------------------------------- #
# BPEMerge dataclass
# ---------------------------------------------------------------------- #


class TestBPEMerge:
    """The :class:`BPEMerge` value object."""

    def test_to_dict_round_trip(self) -> None:
        m = BPEMerge(pair=("a", "b"), rank=3, new_token="ab")
        d = m.to_dict()
        assert d == {"pair": ["a", "b"], "rank": 3, "new_token": "ab"}
        m2 = BPEMerge.from_dict(d)
        assert m == m2

    def test_from_dict_invalid_pair_not_list(self) -> None:
        with pytest.raises(ValueError, match="pair"):
            BPEMerge.from_dict({"pair": "ab", "rank": 0, "new_token": "ab"})

    def test_from_dict_invalid_pair_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="pair"):
            BPEMerge.from_dict({"pair": ["a"], "rank": 0, "new_token": "a"})

    def test_from_dict_non_string_components(self) -> None:
        with pytest.raises(ValueError, match="pair components"):
            BPEMerge.from_dict({"pair": [1, 2], "rank": 0, "new_token": "12"})

    def test_from_dict_invalid_rank(self) -> None:
        with pytest.raises(ValueError, match="rank"):
            BPEMerge.from_dict({"pair": ["a", "b"], "rank": "zero", "new_token": "ab"})

    def test_from_dict_rank_bool_rejected(self) -> None:
        """``bool`` is a subclass of ``int`` in Python — explicit guard."""
        with pytest.raises(ValueError, match="rank"):
            BPEMerge.from_dict({"pair": ["a", "b"], "rank": True, "new_token": "ab"})

    def test_from_dict_invalid_new_token(self) -> None:
        with pytest.raises(ValueError, match="new_token"):
            BPEMerge.from_dict({"pair": ["a", "b"], "rank": 0, "new_token": 42})


# ---------------------------------------------------------------------- #
# Decode coverage — UNK, non-eow tokens
# ---------------------------------------------------------------------- #


class TestDecodeCoverage:
    """Cover the branches in :meth:`BPETokenizer.decode` that the round-trip
    tests don't reach (UNK, mid-word non-eow tokens)."""

    def test_decode_unk_token_emits_replacement_char(self) -> None:
        """The literal ``<unk>`` token decodes to U+FFFD."""
        tk = train_bpe("hello world " * 50, vocab_size=50)
        # Force an UNK id into the stream by encoding a non-corpus char.
        ids = tk.encode("\u4e2d")  # Chinese — never seen
        assert tk.vocab[UNK] in ids
        decoded = tk.decode(ids)
        # Each UNK decodes to \ufffd.
        assert "\ufffd" in decoded

    def test_decode_mixed_eow_and_non_eow_tokens(self) -> None:
        """EOW-suffixed tokens flush as word boundaries when decoding.

        ``train_bpe`` never produces a standalone ``</w>`` token (the marker
        only lives as a suffix of merged word tokens like ``"low</w>"``), so
        this exercises the real code path: an EOW-suffixed token flushes the
        pending char buffer into a completed word.
        """
        tk = train_bpe("low low low lower lower", vocab_size=30)
        low_eow_id = tk.vocab["low</w>"]
        # Two completed words back-to-back → joined with a single space.
        assert tk.decode([low_eow_id, low_eow_id]) == "low low"
        # A bare char followed by an EOW-suffixed token: the pending char
        # is flushed into the SAME word as the EOW stem (chars concatenate
        # into the word, no space inserted), producing "l" + "low" = "llow".
        # This documents decode's word-assembly contract: an EOW-suffixed
        # token completes the current word, it does not start a new one.
        char_id = tk.vocab["l"]
        assert tk.decode([char_id, low_eow_id]) == "llow"
        # Two EOW-suffixed tokens that both flush empty stems collapse
        # to a single empty word — assert no spurious spaces leak in.
        # Inject a literal "</w>" id to cover the empty-stem branch of
        # ``decode`` (unreachable from ``train_bpe`` output alone).
        tk.vocab["</w>"] = len(tk.vocab)
        tk._id_to_token[tk.vocab["</w>"]] = "</w>"
        assert tk.decode([tk.vocab["</w>"], tk.vocab["</w>"]]) == ""


# ---------------------------------------------------------------------- #
# Load edge cases
# ---------------------------------------------------------------------- #


class TestLoadEdgeCases:
    """Defensive paths in :meth:`BPETokenizer.load`."""

    def test_load_invalid_merges_type(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"vocab": {"a": 1}, "merges": "not a list"}))
        with pytest.raises(ValueError, match="Invalid merges field"):
            BPETokenizer.load(path)

    def test_load_invalid_merge_entry(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"vocab": {"a": 1}, "merges": ["not a dict"]}))
        with pytest.raises(ValueError, match="Invalid merge entry"):
            BPETokenizer.load(path)


# ---------------------------------------------------------------------- #
# Training edge cases
# ---------------------------------------------------------------------- #


class TestTrainEdgeCases:
    """Defensive paths in :func:`train_bpe`."""

    def test_whitespace_corpus_returns_no_merges(self) -> None:
        """A corpus with only whitespace yields no merges (no word pairs)."""
        tk = train_bpe("   ", vocab_size=20)
        assert tk.merges == []
        # All four specials still present.
        for tok in SPECIAL_TOKENS:
            assert tok in tk.vocab

    def test_all_base_chars_covered(self) -> None:
        """Every distinct char ends up in the base vocabulary."""
        corpus = "abc 123 !@# xyz"
        tk = train_bpe(corpus, vocab_size=200)
        for ch in set(corpus):
            assert ch in tk.vocab
