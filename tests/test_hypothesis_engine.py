"""Tests for the Autonomous Hypothesis Engine."""

from cds.hypothesis import (
    Domain,
    EvaluationData,
    HypothesisEvaluator,
    HypothesisStatus,
    generate_hypotheses,
)


def test_hypothesis_evaluation_t_test() -> None:
    hypos = generate_hypotheses("Do plants grow faster with music?", Domain.GENERAL_SCIENCE, n=1)
    hypo = hypos[0]

    # Simulate data: Group A (no music), Group B (music)
    # Group B has significantly higher mean
    data: EvaluationData = {
        "groups": [[10.1, 12.2, 11.5, 10.8, 11.0], [15.5, 16.2, 14.8, 17.0, 15.9]]
    }

    evaluator = HypothesisEvaluator(alpha=0.05)
    result = evaluator.evaluate(hypo, data)

    assert result.test_name == "Two-sample t-test"
    assert result.is_significant
    assert hypo.status == HypothesisStatus.VALIDATED
    assert result.p_value < 0.05


def test_hypothesis_evaluation_anova() -> None:
    hypos = generate_hypotheses("Effect of 3 different fertilizers", Domain.GENERAL_SCIENCE, n=1)
    hypo = hypos[0]

    # 3 groups
    data: EvaluationData = {"groups": [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]]}

    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(hypo, data)

    assert result.test_name == "One-way ANOVA"
    assert result.is_significant
    assert hypo.status == HypothesisStatus.VALIDATED


def test_hypothesis_evaluation_rejected() -> None:
    hypos = generate_hypotheses("Testing null effect", Domain.GENERAL_SCIENCE, n=1)
    hypo = hypos[0]

    # Similar means
    data: EvaluationData = {"groups": [[10.0, 10.1, 10.2], [10.0, 10.1, 9.9]]}

    evaluator = HypothesisEvaluator()
    result = evaluator.evaluate(hypo, data)

    assert not result.is_significant
    assert hypo.status == HypothesisStatus.REJECTED
