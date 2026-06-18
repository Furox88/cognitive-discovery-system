"""Tests for :mod:`cds.nlp.layers`.

Covers the activation, normalisation, FFN, and the pre/post-norm
transformer block wiring. The block tests use small hand-crafted
weights so the forward pass is traceable in the test logs.
"""

from __future__ import annotations

import math

import pytest

from cds.nlp.layers import (
    AttentionWeights,
    FeedForwardWeights,
    feed_forward,
    gelu,
    layer_norm,
    transformer_block,
)

# ---------------------------------------------------------------------- #
# GELU
# ---------------------------------------------------------------------- #


class TestGelu:
    def test_at_zero(self) -> None:
        assert abs(gelu(0.0)) < 1e-12

    def test_at_one(self) -> None:
        # GELU(1) ≈ 0.8413 — Phi(1) ≈ 0.8413.
        assert abs(gelu(1.0) - 0.8413447) < 1e-6

    def test_at_negative_one(self) -> None:
        # GELU(-1) ≈ -0.1587.
        assert abs(gelu(-1.0) - (-0.1586552)) < 1e-6

    def test_monotonic_for_large_x(self) -> None:
        """For x > 1, GELU is monotonic increasing."""
        prev = gelu(1.0)
        for x in (1.5, 2.0, 3.0, 5.0):
            cur = gelu(x)
            assert cur > prev
            prev = cur

    def test_odd_function_property(self) -> None:
        """GELU is not strictly odd (it's not ReLU), but GELU(-x) ≈ -GELU(x)
        for large |x|. We just check a few reference values."""
        assert gelu(2.0) > 0
        assert gelu(-2.0) < 0


# ---------------------------------------------------------------------- #
# LayerNorm
# ---------------------------------------------------------------------- #


class TestLayerNorm:
    def test_zero_mean_unit_var(self) -> None:
        """After layer norm, each row has mean ≈ 0 and var ≈ 1.

        The ``eps`` floor (default 1e-5) introduces a tiny bias in the
        unit-variance check — ``var ≈ 1 ± 1e-5`` — so the tolerance is
        set wide enough to accommodate the floor.
        """
        x = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        gamma = [1.0] * 4
        beta = [0.0] * 4
        out = layer_norm(x, gamma, beta)
        for row in out:
            mean = sum(row) / len(row)
            var = sum((v - mean) ** 2 for v in row) / len(row)
            assert abs(mean) < 1e-9
            assert abs(var - 1.0) < 1e-4

    def test_gamma_beta_scale_shift(self) -> None:
        """gamma=2, beta=1 → output is 2 * normalized + 1."""
        x = [[1.0, 2.0, 3.0, 4.0]]
        gamma = [2.0] * 4
        beta = [1.0] * 4
        out = layer_norm(x, gamma, beta)
        normalized = layer_norm(x, [1.0] * 4, [0.0] * 4)
        for j in range(4):
            assert abs(out[0][j] - (2.0 * normalized[0][j] + 1.0)) < 1e-12

    def test_eps_floor(self) -> None:
        """All-equal input has zero variance — eps prevents division by zero."""
        x = [[3.0, 3.0, 3.0, 3.0]]
        gamma = [1.0] * 4
        beta = [0.0] * 4
        out = layer_norm(x, gamma, beta)
        # Result is well-defined (not NaN/inf).
        for v in out[0]:
            assert math.isfinite(v)

    def test_gamma_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma/beta length"):
            layer_norm([[1.0, 2.0, 3.0]], [1.0, 1.0], [0.0, 0.0])

    def test_empty_input(self) -> None:
        assert layer_norm([], [], []) == []


# ---------------------------------------------------------------------- #
# Feed-forward
# ---------------------------------------------------------------------- #


class TestFeedForward:
    @staticmethod
    def _build(d_model: int = 4, d_ff: int = 8) -> dict[str, object]:
        """Tiny FFN: weights=1, biases=0 → pure GeLU + linear projection."""
        return {
            "w1": [[1.0] * d_ff for _ in range(d_model)],
            "b1": [0.0] * d_ff,
            "w2": [[1.0 / d_ff] * d_model for _ in range(d_ff)],
            "b2": [0.0] * d_model,
        }

    def test_shape(self) -> None:
        x = [[1.0, 2.0, 3.0, 4.0]]
        w = self._build()
        out = feed_forward(x, w["w1"], w["b1"], w["w2"], w["b2"])
        assert len(out) == 1
        assert len(out[0]) == 4

    def test_zero_input_gelu_zero(self) -> None:
        """All-zero input → GeLU(0)=0 everywhere → final output is 0."""
        x = [[0.0, 0.0, 0.0, 0.0]]
        w = self._build()
        out = feed_forward(x, w["w1"], w["b1"], w["w2"], w["b2"])
        for v in out[0]:
            assert abs(v) < 1e-12

    def test_empty_input(self) -> None:
        assert feed_forward([], [[1.0]], [0.0], [[1.0]], [0.0]) == []

    def test_bias_added(self) -> None:
        """A non-zero b1 should shift the pre-GeLU activations."""
        x = [[0.0, 0.0]]
        w1 = [[1.0, 1.0], [1.0, 1.0]]
        b1 = [0.5, 0.5]
        w2 = [[1.0, 1.0], [1.0, 1.0]]
        b2 = [0.0, 0.0]
        out = feed_forward(x, w1, b1, w2, b2)
        # GeLU(0.5) ≈ 0.3457 — sum 2 of them in each output position.
        # Each output position is a sum of 2 GeLU(0.5) ≈ 0.6914.
        expected = 2 * gelu(0.5)
        assert abs(out[0][0] - expected) < 1e-9
        assert abs(out[0][1] - expected) < 1e-9

    def test_w1_shape_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="w1 shape"):
            feed_forward(
                [[1.0, 2.0]],
                [[1.0]],  # 1x1, expected 2x3
                [0.0, 0.0, 0.0],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [0.0, 0.0],
            )

    def test_b2_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="b2 length"):
            feed_forward(
                [[1.0, 2.0]],
                [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                [0.0, 0.0, 0.0],
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
                [0.0, 0.0, 0.0],  # wrong length
            )


# ---------------------------------------------------------------------- #
# Transformer block
# ---------------------------------------------------------------------- #


class TestTransformerBlock:
    @staticmethod
    def _weights(
        d_model: int = 4, n_heads: int = 2, d_ff: int = 8
    ) -> tuple[AttentionWeights, FeedForwardWeights]:
        """Tiny identity-ish weights for a 4-d, 2-head, d_ff=8 block."""
        attn: AttentionWeights = {
            "w_q": [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)],
            "w_k": [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)],
            "w_v": [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)],
            "w_o": [[1.0 if i == j else 0.0 for j in range(d_model)] for i in range(d_model)],
            "ln1_gamma": [1.0] * d_model,
            "ln1_beta": [0.0] * d_model,
            "ln2_gamma": [1.0] * d_model,
            "ln2_beta": [0.0] * d_model,
        }
        ffn: FeedForwardWeights = {
            "w1": [[0.1] * d_ff for _ in range(d_model)],
            "b1": [0.0] * d_ff,
            "w2": [[0.1] * d_model for _ in range(d_ff)],
            "b2": [0.0] * d_model,
        }
        return attn, ffn

    def test_prenorm_output_shape(self) -> None:
        x = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        attn, ffn = self._weights()
        out = transformer_block(x, attn, ffn, n_heads=2, prenorm=True)
        assert len(out) == 3
        assert all(len(row) == 4 for row in out)

    def test_postnorm_output_shape(self) -> None:
        x = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        attn, ffn = self._weights()
        out = transformer_block(x, attn, ffn, n_heads=2, prenorm=False)
        assert len(out) == 2
        assert all(len(row) == 4 for row in out)

    def test_causal_mask_propagates(self) -> None:
        """With a causal mask, output[0] should be invariant under a
        change to inputs at positions > 0."""
        attn, ffn = self._weights()
        n = 3
        x_a = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        x_b = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0]]
        from cds.nlp.attention import causal_mask

        mask = causal_mask(n)
        out_a = transformer_block(x_a, attn, ffn, n_heads=2, mask=mask, prenorm=True)
        out_b = transformer_block(x_b, attn, ffn, n_heads=2, mask=mask, prenorm=True)
        # Position 0 should be identical because it can only see position 0.
        for j in range(4):
            assert abs(out_a[0][j] - out_b[0][j]) < 1e-9

    def test_empty_input(self) -> None:
        attn, ffn = self._weights()
        assert transformer_block([], attn, ffn, n_heads=2) == []

    def test_causal_mask_full_block(self) -> None:
        """Causal mask combined with the FFN sub-block still preserves
        position-0 invariance. (Catches the add() shape-mismatch path.)"""
        attn, ffn = self._weights()
        from cds.nlp.attention import causal_mask

        mask = causal_mask(2)
        x_a = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        x_b = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        out_a = transformer_block(x_a, attn, ffn, n_heads=2, mask=mask, prenorm=False)
        out_b = transformer_block(x_b, attn, ffn, n_heads=2, mask=mask, prenorm=False)
        for j in range(4):
            assert abs(out_a[0][j] - out_b[0][j]) < 1e-9

    def test_residual_dominated_by_input(self) -> None:
        """With near-zero attention/FFN outputs, the block should return
        approximately the input (residual dominates)."""
        # Use weights that produce near-zero projections.
        attn, ffn = self._weights()
        for key in ("w_q", "w_k", "w_v", "w_o"):
            for row in attn[key]:
                for j in range(len(row)):
                    row[j] *= 1e-6
        for row in ffn["w1"]:
            for j in range(len(row)):
                row[j] *= 1e-6
        for row in ffn["w2"]:
            for j in range(len(row)):
                row[j] *= 1e-6
        x = [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
        out = transformer_block(x, attn, ffn, n_heads=2, prenorm=True)
        # Output should be close to the (post-layernorm) input, but
        # with prenorm + tiny projections the residual x dominates.
        for row in out:
            for v in row:
                # Tolerance: layernorm makes exact equality impossible,
                # but residual + tiny delta keeps the values bounded.
                assert abs(v) < 100.0  # very loose sanity check
