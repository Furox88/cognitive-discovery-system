"""Tests for :mod:`cds.nlp.data` and :mod:`cds.nlp.model`.

Covers the Shakespeare corpus loader and the MiniGPT shape /
forward pass / sample smoke tests. We don't train a real model here
— the demo in ``examples/nlp_mini_gpt_demo.py`` does that — but we
verify that the forward pass produces a finite logits vector of
the right size, that the loss decreases over a few gradient steps,
and that sample() returns a sequence of the right length.
"""

from __future__ import annotations

import pytest

from cds.nlp.data import TEXT, decode, encode, vocab_size
from cds.nlp.model import MiniGPT
from cds.nlp.optim import Adam
from cds.nlp.training import train_step

# ---------------------------------------------------------------------- #
# Shakespeare corpus
# ---------------------------------------------------------------------- #


class TestShakespeareCorpus:
    def test_text_nonempty(self) -> None:
        assert len(TEXT) > 1000
        assert "All the world's a stage" in TEXT

    def test_vocab_size_is_reasonable(self) -> None:
        # 26 letters + space + punctuation + newline ≈ 40-50.
        assert 30 < vocab_size < 80

    def test_encode_decode_round_trip(self) -> None:
        ids = encode(TEXT)
        assert isinstance(ids, list)
        assert all(isinstance(i, int) for i in ids)
        assert all(0 <= i < vocab_size for i in ids)
        assert decode(ids) == TEXT

    def test_encode_partial_string(self) -> None:
        # Encoding a substring present in the corpus should round-trip.
        sample = "All the"
        ids = encode(sample)
        assert decode(ids) == sample

    def test_decode_handles_empty_list(self) -> None:
        assert decode([]) == ""


# ---------------------------------------------------------------------- #
# MiniGPT
# ---------------------------------------------------------------------- #


class TestMiniGPT:
    def test_construction(self) -> None:
        model = MiniGPT(vocab_size=10, d_model=8, n_heads=2, d_ff=16, max_len=8)
        assert model.vocab_size == 10
        assert model.d_model == 8
        assert model.n_heads == 2
        assert model.d_ff == 16
        assert model.max_len == 8

    def test_d_model_must_divide_n_heads(self) -> None:
        with pytest.raises(ValueError, match="divisible"):
            MiniGPT(vocab_size=10, d_model=7, n_heads=2)

    def test_d_ff_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="d_ff"):
            MiniGPT(vocab_size=10, d_model=8, d_ff=0)

    def test_max_len_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="max_len"):
            MiniGPT(vocab_size=10, max_len=0)

    def test_parameters_returns_trainable_tensors(self) -> None:
        model = MiniGPT(vocab_size=4, d_model=4, n_heads=2, d_ff=4, max_len=4)
        params = model.parameters()
        assert len(params) > 0
        assert all(p.requires_grad for p in params)

    def test_forward_returns_vocab_logits(self) -> None:
        model = MiniGPT(vocab_size=4, d_model=4, n_heads=2, d_ff=4, max_len=4)
        ids = [0, 1, 2]
        logits = model.forward(ids)
        assert len(logits) == 4
        for li in logits:
            # Each logit is a Tensor with the autograd graph attached.
            assert li.requires_grad is True

    def test_forward_too_long_raises(self) -> None:
        model = MiniGPT(vocab_size=4, d_model=4, n_heads=2, d_ff=4, max_len=4)
        with pytest.raises(ValueError, match="exceeds max_len"):
            model.forward([0, 1, 2, 3, 4, 5])

    def test_forward_empty_returns_empty(self) -> None:
        model = MiniGPT(vocab_size=4, d_model=4, n_heads=2, d_ff=4, max_len=4)
        assert model.forward([]) == []

    def test_sample_returns_correct_length(self) -> None:
        model = MiniGPT(vocab_size=4, d_model=4, n_heads=2, d_ff=4, max_len=4)
        prompt = [0, 1]
        out = model.sample(prompt, n_tokens=3)
        assert len(out) == 5  # prompt (2) + generated (3)

    def test_sample_zero_tokens_returns_prompt(self) -> None:
        model = MiniGPT(vocab_size=4, d_model=4, n_heads=2, d_ff=4, max_len=4)
        prompt = [0, 1, 2]
        out = model.sample(prompt, n_tokens=0)
        assert out == prompt

    def test_loss_decreases_over_few_steps(self) -> None:
        """A handful of train_steps on the Shakespeare excerpt should
        bring the loss down. (We don't expect a perfect model — just
        that gradients flow and the optimiser updates parameters.)"""
        model = MiniGPT(vocab_size=vocab_size, d_model=16, n_heads=2, d_ff=32, max_len=64)
        # Use a 32-char window so the input fits well within max_len.
        ids = encode(TEXT[:33])
        opt = Adam(params=list(model.parameters()), lr=0.01)
        losses: list[float] = []
        for _ in range(15):
            loss = train_step(model.forward, ids[:-1], ids[-1], opt)
            losses.append(loss)
        # The loss should trend downward even on 15 steps.
        assert losses[-1] < losses[0]
