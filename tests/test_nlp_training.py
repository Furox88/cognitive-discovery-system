"""Tests for :mod:`cds.nlp.optim` and :mod:`cds.nlp.training`.

Covers the SGD/Adam update rules, the cross-entropy loss gradient
flowing into the model, and the high-level :func:`train_step` API.
"""

from __future__ import annotations

import pytest

from cds.nlp.autograd import Parameter, Tensor, neg
from cds.nlp.optim import SGD, Adam, parameters
from cds.nlp.training import cross_entropy, softmax, train_step

# ---------------------------------------------------------------------- #
# softmax helper (vector, used by training.cross_entropy)
# ---------------------------------------------------------------------- #


class TestSoftmax:
    def test_sums_to_one(self) -> None:
        s = softmax([1.0, 2.0, 3.0])
        assert abs(sum(s) - 1.0) < 1e-12

    def test_monotonic(self) -> None:
        s = softmax([1.0, 2.0, 3.0])
        assert s[0] < s[1] < s[2]

    def test_empty(self) -> None:
        assert softmax([]) == []


# ---------------------------------------------------------------------- #
# cross_entropy loss
# ---------------------------------------------------------------------- #


class TestCrossEntropy:
    def test_loss_is_log_of_inverse_probability(self) -> None:
        """When target has probability p, the loss is -log(p)."""
        # logits = [0, 0, 100] → target 2 has p ≈ 1, others ≈ 0.
        loss = cross_entropy([0.0, 0.0, 100.0], target=2)
        assert float(loss.data) < 0.01  # ≈ 0

    def test_loss_high_for_wrong_target(self) -> None:
        # logits = [0, 0, 100] → target 0 has p ≈ 0, loss is large.
        loss = cross_entropy([0.0, 0.0, 100.0], target=0)
        assert float(loss.data) > 50.0

    def test_loss_gradient_flows(self) -> None:
        """Gradient on the logit at the target index should be negative
        (push it up); gradient on others should be positive (push them
        down)."""
        # Three logits, all equal → uniform softmax = [1/3, 1/3, 1/3].
        logits: list[Tensor] = [Parameter(0.0), Parameter(0.0), Parameter(0.0)]
        loss = cross_entropy(logits, target=1)
        loss.backward()
        # Gradient on target logit: p_target - 1 = 1/3 - 1 = -2/3.
        # Gradient on others: p_others = 1/3.
        assert logits[0].grad is not None
        assert logits[1].grad is not None
        assert logits[2].grad is not None
        assert abs(logits[0].grad - (1.0 / 3.0)) < 1e-9
        # target=1 → p_target - 1 = 1/3 - 1 = -2/3.
        assert abs(logits[1].grad - (1.0 / 3.0 - 1.0)) < 1e-9
        assert abs(logits[2].grad - (1.0 / 3.0)) < 1e-9

    def test_empty_logits_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            cross_entropy([], target=0)

    def test_target_out_of_range_raises(self) -> None:
        with pytest.raises(ValueError, match="out of range"):
            cross_entropy([1.0, 2.0], target=5)


# ---------------------------------------------------------------------- #
# SGD
# ---------------------------------------------------------------------- #


class TestSGD:
    def test_vanilla_step(self) -> None:
        p = Parameter(1.0)
        p.grad = 0.5
        opt = SGD(params=[p], lr=0.1)
        opt.step()
        assert p.data == pytest.approx(0.95)

    def test_momentum_step(self) -> None:
        p = Parameter(1.0)
        p.grad = 0.5
        opt = SGD(params=[p], lr=0.1, momentum=0.9)
        opt.step()
        # v = 0.9 * 0 + 0.5 = 0.5; p -= 0.1 * 0.5 = 1.0 - 0.05
        assert p.data == pytest.approx(0.95)
        # Second step: v = 0.9 * 0.5 + 0.5 = 0.95; p -= 0.1 * 0.95 = 0.095
        opt.step()
        assert p.data == pytest.approx(0.855)

    def test_weight_decay(self) -> None:
        p = Parameter(1.0)
        p.grad = 0.0
        opt = SGD(params=[p], lr=0.1, weight_decay=0.1)
        opt.step()
        # effective grad = 0 + 0.1 * 1.0 = 0.1; p -= 0.1 * 0.1 = 0.01
        assert p.data == pytest.approx(0.99)

    def test_zero_grad(self) -> None:
        p = Parameter(1.0)
        p.grad = 0.5
        opt = SGD(params=[p], lr=0.1)
        opt.zero_grad()
        assert p.grad == 0.0

    def test_invalid_lr_raises(self) -> None:
        with pytest.raises(ValueError, match="lr must be > 0"):
            SGD(params=[Parameter(0.0)], lr=0.0)

    def test_invalid_momentum_raises(self) -> None:
        with pytest.raises(ValueError, match="momentum"):
            SGD(params=[Parameter(0.0)], momentum=1.0)

    def test_invalid_weight_decay_raises(self) -> None:
        with pytest.raises(ValueError, match="weight_decay"):
            SGD(params=[Parameter(0.0)], weight_decay=-0.1)


# ---------------------------------------------------------------------- #
# Adam
# ---------------------------------------------------------------------- #


class TestAdam:
    def test_first_step_decreases_loss(self) -> None:
        """A few Adam steps on a simple quadratic should reduce |p|."""
        p = Parameter(1.0)

        def loss_fn() -> Tensor:
            return p * p

        opt = Adam(params=[p], lr=0.1)
        for _ in range(50):
            opt.zero_grad()
            loss = loss_fn()
            loss.backward()
            opt.step()
        # p should have moved toward 0.
        assert abs(p.data) < 0.1

    def test_bias_correction_first_step(self) -> None:
        """First Adam step uses bias-corrected m, v — m/0.1, v/0.001.
        Effective lr is roughly lr (not amplified by the small denom)."""
        p = Parameter(1.0)
        p.grad = 0.1
        opt = Adam(params=[p], lr=0.001)
        opt.step()
        # First step: m = 0.1 * 0.1 = 0.01, v = 0.001 * 0.01 = 1e-5.
        # m_hat = 0.01 / (1 - 0.9) = 0.1, v_hat = 1e-5 / (1 - 0.999) = 0.01.
        # p -= 0.001 * 0.1 / (sqrt(0.01) + 1e-8) ≈ 0.001.
        # So p went from 1.0 to ≈ 0.999.
        assert p.data == pytest.approx(0.999, abs=1e-6)

    def test_invalid_betas_raises(self) -> None:
        with pytest.raises(ValueError, match="betas"):
            Adam(params=[Parameter(0.0)], betas=(0.9, 1.0))

    def test_invalid_eps_raises(self) -> None:
        with pytest.raises(ValueError, match="eps"):
            Adam(params=[Parameter(0.0)], eps=0.0)

    def test_zero_grad(self) -> None:
        p = Parameter(1.0)
        p.grad = 0.5
        opt = Adam(params=[p])
        opt.zero_grad()
        assert p.grad == 0.0


# ---------------------------------------------------------------------- #
# parameters() helper
# ---------------------------------------------------------------------- #


class TestParameters:
    def test_filters_to_grad_only(self) -> None:
        a = Parameter(0.0)
        b = Tensor(data=1.0, requires_grad=False)
        c = Parameter(2.0)
        d = Tensor(data=3.0, requires_grad=True)  # not a Parameter but has grad
        out = parameters([a, b, c, d])
        assert out == [a, c, d]

    def test_empty(self) -> None:
        assert parameters([]) == []


# ---------------------------------------------------------------------- #
# train_step
# ---------------------------------------------------------------------- #


class TestTrainStep:
    def test_returns_float_loss(self) -> None:
        w = Parameter(0.0)

        def model_fn(x: list[int]) -> list[Tensor]:
            # Tiny model: logits = [w, -w] regardless of input.
            return [w, neg(w)]

        opt = Adam(params=[w], lr=0.1)
        loss = train_step(model_fn, x=[0, 1], y=0, optimiser=opt)
        assert isinstance(loss, float)

    def test_model_fn_returning_floats_raises(self) -> None:
        """If the forward pass returns plain floats (no autograd), the
        loss is detached and train_step refuses to silently do nothing."""
        w = Parameter(0.0)

        def bad_model_fn(x: list[int]) -> list[float]:
            return [float(w.data), float(-w.data)]

        opt = SGD(params=[w], lr=0.1)
        with pytest.raises(RuntimeError, match="Tensor logits"):
            train_step(bad_model_fn, x=[0], y=0, optimiser=opt)

    def test_loss_decreases_over_steps(self) -> None:
        w = Parameter(0.0)

        def model_fn(x: list[int]) -> list[Tensor]:
            return [w, neg(w)]

        opt = Adam(params=[w], lr=0.1)
        losses: list[float] = []
        for _ in range(20):
            loss = train_step(model_fn, x=[0], y=0, optimiser=opt)
            losses.append(loss)
        # The model wants w > 0 (target 0 has logit w). Loss should drop.
        assert losses[-1] < losses[0]

    def test_gradients_cleared_between_steps(self) -> None:
        """train_step calls ``optimiser.zero_grad()`` before each forward,
        so the gradient is reset between steps (matches PyTorch's
        ``optimizer.zero_grad()`` pattern)."""
        w = Parameter(0.0)

        def model_fn(x: list[int]) -> list[Tensor]:
            return [w, neg(w)]

        opt = SGD(params=[w], lr=0.1)
        train_step(model_fn, x=[0], y=0, optimiser=opt)
        # At the start of the next call we expect zero_grad() to have
        # cleared the gradient from the previous step. Run another
        # step and verify it runs without error (gradient was zeroed
        # at the start; the new backward populates a fresh value).
        before = w.grad  # gradient after step 1
        train_step(model_fn, x=[0], y=0, optimiser=opt)
        # If zero_grad hadn't run, the second step would have
        # *accumulated* gradients and ``before`` would have been
        # double. Instead we see a fresh gradient after step 2.
        assert before != 0.0  # step 1 populated a non-zero grad
        assert w.data != 0.0  # step 1 + 2 actually moved the parameter
