"""Tests for :mod:`cds.nlp.embed`.

Covers token / positional embedding shapes, mathematical correctness of
the sinusoidal encoding, determinism of the seeded RNG, and the
``add_positional`` helper.
"""

from __future__ import annotations

import math

import pytest

from cds.nlp.embed import PositionalEncoding, TokenEmbedding, add_positional

# ---------------------------------------------------------------------- #
# Fixtures
# ---------------------------------------------------------------------- #


@pytest.fixture
def vocab_size() -> int:
    return 64


@pytest.fixture
def d_model() -> int:
    return 32


@pytest.fixture
def token_table(vocab_size: int, d_model: int) -> TokenEmbedding:
    return TokenEmbedding(vocab_size=vocab_size, d_model=d_model)


@pytest.fixture
def positional(d_model: int) -> PositionalEncoding:
    return PositionalEncoding(max_len=128, d_model=d_model)


# ---------------------------------------------------------------------- #
# TokenEmbedding
# ---------------------------------------------------------------------- #


class TestTokenEmbedding:
    """Token lookup table correctness."""

    def test_shape_property(
        self, token_table: TokenEmbedding, vocab_size: int, d_model: int
    ) -> None:
        assert token_table.shape == (vocab_size, d_model)

    def test_matrix_dimensions(
        self, token_table: TokenEmbedding, vocab_size: int, d_model: int
    ) -> None:
        assert len(token_table.matrix) == vocab_size
        for row in token_table.matrix:
            assert len(row) == d_model

    def test_forward_returns_one_row_per_id(
        self, token_table: TokenEmbedding, d_model: int
    ) -> None:
        out = token_table.forward([0, 1, 2])
        assert len(out) == 3
        for row in out:
            assert len(row) == d_model

    def test_forward_returns_copy(self, token_table: TokenEmbedding) -> None:
        """Mutating the returned embedding must not poison the table."""
        out = token_table.forward([0])
        out[0][0] = 999.0
        assert token_table.matrix[0][0] != 999.0

    def test_negative_id_raises(self, token_table: TokenEmbedding) -> None:
        with pytest.raises(IndexError, match="out of range"):
            token_table.forward([-1])

    def test_id_above_vocab_raises(self, token_table: TokenEmbedding, vocab_size: int) -> None:
        with pytest.raises(IndexError, match="out of range"):
            token_table.forward([vocab_size])

    def test_set_value_replaces_row(self, token_table: TokenEmbedding, d_model: int) -> None:
        new_row = [float(i) for i in range(d_model)]
        token_table.set_value(5, new_row)
        assert token_table.matrix[5] == new_row

    def test_set_value_wrong_length_raises(self, token_table: TokenEmbedding) -> None:
        with pytest.raises(ValueError, match="values length"):
            token_table.set_value(0, [1.0, 2.0])

    def test_set_value_defensive_copy(self, token_table: TokenEmbedding, d_model: int) -> None:
        """Overwriting via set_value must store a copy."""
        row = [0.0] * d_model
        token_table.set_value(3, row)
        row[0] = 999.0
        assert token_table.matrix[3][0] == 0.0

    def test_init_deterministic(self, vocab_size: int, d_model: int) -> None:
        """Same seed → same matrix."""
        a = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        b = TokenEmbedding(vocab_size=vocab_size, d_model=d_model)
        assert a.matrix == b.matrix

    def test_invalid_vocab_size_raises(self, d_model: int) -> None:
        with pytest.raises(ValueError, match="vocab_size"):
            TokenEmbedding(vocab_size=0, d_model=d_model)

    def test_invalid_d_model_raises(self, vocab_size: int) -> None:
        with pytest.raises(ValueError, match="d_model"):
            TokenEmbedding(vocab_size=vocab_size, d_model=0)

    def test_weights_bounded(self, token_table: TokenEmbedding, d_model: int) -> None:
        """Init bound is 1/sqrt(d_model)."""
        bound = 1.0 / math.sqrt(d_model) + 1e-9
        for row in token_table.matrix:
            for v in row:
                assert abs(v) <= bound


# ---------------------------------------------------------------------- #
# PositionalEncoding
# ---------------------------------------------------------------------- #


class TestPositionalEncoding:
    """Sinusoidal position embedding."""

    def test_shape_property(self, positional: PositionalEncoding, d_model: int) -> None:
        assert positional.shape == (128, d_model)

    def test_matrix_dimensions(self, positional: PositionalEncoding, d_model: int) -> None:
        assert len(positional.matrix) == 128
        for row in positional.matrix:
            assert len(row) == d_model

    def test_position_zero_is_sin_zero_for_even_indices(
        self, positional: PositionalEncoding
    ) -> None:
        """sin(0) = 0 for even indices in row 0."""
        for i in range(0, positional.d_model, 2):
            assert abs(positional.matrix[0][i]) < 1e-12

    def test_position_zero_is_cos_zero_for_odd_indices(
        self, positional: PositionalEncoding
    ) -> None:
        """cos(0) = 1 for odd indices in row 0."""
        for i in range(1, positional.d_model, 2):
            assert abs(positional.matrix[0][i] - 1.0) < 1e-12

    def test_first_pair_is_period_one(self, positional: PositionalEncoding) -> None:
        """i=0,1 → div_term = 10000^0 = 1, so the first dimension
        oscillates with period 2π. Verify pos 0 and pos 1 differ."""
        # pos 0, dim 0 → sin(0) = 0
        # pos 1, dim 0 → sin(1)
        assert abs(positional.matrix[0][0]) < 1e-12
        assert abs(positional.matrix[1][0] - math.sin(1.0)) < 1e-12

    def test_high_index_period_grows(self, positional: PositionalEncoding) -> None:
        """The highest-index pair oscillates slowly: pos=0 and pos=100 in
        the last two columns should be very close (period ~2π·10000)."""
        last = positional.d_model - 2
        # With d_model=32 the last index pair has period 2π·10000^(30/32) ≈ 28116.
        # pos=100 covers only ~2% of that, so |pos=0 − pos=100| ≪ 1.
        assert abs(positional.matrix[0][last] - positional.matrix[100][last]) < 0.05

    def test_forward_returns_first_n_rows(self, positional: PositionalEncoding) -> None:
        rows = positional.forward(5)
        assert len(rows) == 5
        for i, row in enumerate(rows):
            assert row == positional.matrix[i]

    def test_forward_zero_length_ok(self, positional: PositionalEncoding) -> None:
        assert positional.forward(0) == []

    def test_forward_negative_raises(self, positional: PositionalEncoding) -> None:
        with pytest.raises(ValueError, match="length"):
            positional.forward(-1)

    def test_forward_too_long_raises(self, positional: PositionalEncoding) -> None:
        with pytest.raises(ValueError, match="max_len"):
            positional.forward(129)

    def test_invalid_max_len_raises(self, d_model: int) -> None:
        with pytest.raises(ValueError, match="max_len"):
            PositionalEncoding(max_len=0, d_model=d_model)

    def test_invalid_d_model_raises(self) -> None:
        with pytest.raises(ValueError, match="d_model"):
            PositionalEncoding(max_len=10, d_model=0)

    def test_unique_position_patterns(self, positional: PositionalEncoding) -> None:
        """No two positions should produce identical vectors."""
        rows = [tuple(r) for r in positional.matrix]
        assert len(set(rows)) == len(rows)


# ---------------------------------------------------------------------- #
# add_positional
# ---------------------------------------------------------------------- #


class TestAddPositional:
    """The helper that combines token + position embeddings."""

    def test_empty_token_embeddings_returns_empty(self, positional: PositionalEncoding) -> None:
        assert add_positional([], positional) == []

    def test_shape_preserved(
        self, token_table: TokenEmbedding, positional: PositionalEncoding
    ) -> None:
        tokens = token_table.forward([0, 1, 2])
        out = add_positional(tokens, positional)
        assert len(out) == 3
        assert all(len(row) == positional.d_model for row in out)

    def test_correctness(self, d_model: int) -> None:
        """Manually: out[0][0] = tokens[0][0] + sin(0) = tokens[0][0]."""
        pos = PositionalEncoding(max_len=4, d_model=d_model)
        tokens = [[1.0] * d_model]
        out = add_positional(tokens, pos)
        # Even index in row 0 → sin(0) = 0, so the value stays.
        assert abs(out[0][0] - 1.0) < 1e-12
        # Odd index in row 0 → cos(0) = 1, so the value becomes 2.0.
        assert abs(out[0][1] - 2.0) < 1e-12

    def test_d_model_mismatch_raises(self, token_table: TokenEmbedding) -> None:
        """Token embeddings d_model != positional d_model → ValueError."""
        wrong = PositionalEncoding(max_len=10, d_model=token_table.d_model + 8)
        tokens = token_table.forward([0])
        with pytest.raises(ValueError, match="d_model mismatch"):
            add_positional(tokens, wrong)

    def test_input_not_mutated(
        self, token_table: TokenEmbedding, positional: PositionalEncoding
    ) -> None:
        tokens = token_table.forward([0, 1])
        snapshot = [list(row) for row in tokens]
        add_positional(tokens, positional)
        assert tokens == snapshot
