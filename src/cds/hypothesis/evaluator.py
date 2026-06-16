"""
Evaluator for scientific hypotheses using empirical data.

This module provides the connection between generated hypotheses and
statistical verification, enabling an autonomous discovery loop.

Supported data formats for ``evaluate``:

- ``{"groups": [[...], [...]]}``      -> t-test (2 groups) or ANOVA (3+ groups)
- ``{"one_sample": [...], "popmean": m}`` -> one-sample t-test vs a reference mean
- ``{"paired": ([...], [...])}``       -> paired comparison via two-sample t-test
- ``{"chi_square_gof": {"observed": [...], "expected": [...]}}``
- ``{"chi_square_independence": [[...], ...]}`` (contingency table)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cds.core.models import Hypothesis, HypothesisStatus
from cds.stats.hypothesis_tests import (
    chi_square_gof,
    chi_square_independence,
    one_sample_ttest,
    one_way_anova,
    two_sample_ttest,
)


@dataclass
class EvaluationResult:
    """Detailed result of a hypothesis evaluation."""

    hypothesis_id: str
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    conclusion: str


class HypothesisEvaluator:
    """Autonomous evaluator that matches hypotheses with statistical tests."""

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def _build_result(
        self,
        hypothesis: Hypothesis,
        test_name: str,
        statistic: float,
        p_value: float,
    ) -> EvaluationResult:
        """Format the outcome and update the hypothesis status."""
        is_sig = p_value < self.alpha
        if is_sig:
            conclusion = (
                f"Hypothesis supported at alpha={self.alpha}. "
                f"Significant result found ({test_name})."
            )
            hypothesis.status = HypothesisStatus.VALIDATED
        else:
            conclusion = (
                f"Failed to support hypothesis at alpha={self.alpha}. "
                f"No significant result ({test_name})."
            )
            hypothesis.status = HypothesisStatus.REJECTED
        return EvaluationResult(
            hypothesis_id=hypothesis.id,
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            is_significant=is_sig,
            conclusion=conclusion,
        )

    def compare_groups(
        self,
        hypothesis: Hypothesis,
        groups: list[list[float]],
        labels: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate a hypothesis by comparing multiple numeric groups.

        Uses t-test for 2 groups, ANOVA for more.
        """
        if len(groups) < 2:
            raise ValueError("Evaluation requires at least 2 groups of data.")

        if len(groups) == 2:
            res = two_sample_ttest(groups[0], groups[1])
            test_name = "Two-sample t-test"
        else:
            res = one_way_anova(*groups)
            test_name = "One-way ANOVA"

        return self._build_result(hypothesis, test_name, res.statistic, res.p_value)

    def compare_to_reference(
        self,
        hypothesis: Hypothesis,
        sample: list[float],
        popmean: float,
    ) -> EvaluationResult:
        """One-sample t-test: does the sample differ from a reference mean?"""
        if len(sample) < 2:
            raise ValueError("One-sample evaluation requires at least 2 observations.")
        res = one_sample_ttest(sample, popmean)
        return self._build_result(hypothesis, "One-sample t-test", res.statistic, res.p_value)

    def goodness_of_fit(
        self,
        hypothesis: Hypothesis,
        observed: list[float],
        expected: list[float] | None = None,
    ) -> EvaluationResult:
        """Chi-square goodness-of-fit: observed vs expected category counts.

        If ``expected`` is omitted, a uniform distribution over the categories
        is assumed (all categories equally likely).
        """
        if len(observed) < 2:
            raise ValueError("Goodness-of-fit requires at least 2 categories.")
        if expected is None:
            total = sum(observed)
            n = len(observed)
            expected = [total / n] * n
        res = chi_square_gof(observed, expected)
        return self._build_result(
            hypothesis, "Chi-square goodness-of-fit", res.statistic, res.p_value
        )

    def test_independence(
        self,
        hypothesis: Hypothesis,
        table: list[list[float]],
    ) -> EvaluationResult:
        """Chi-square test of independence on a contingency table."""
        if len(table) < 2 or any(len(row) < 2 for row in table):
            raise ValueError("Independence test requires a 2x2 or larger contingency table.")
        res = chi_square_independence(table)
        return self._build_result(hypothesis, "Chi-square independence", res.statistic, res.p_value)

    def evaluate(self, hypothesis: Hypothesis, data: dict[str, Any]) -> EvaluationResult:
        """General evaluation entry point dispatching on the data format.

        Supported keys (checked in order):

        - ``groups``            : list of numeric groups (t-test / ANOVA)
        - ``one_sample`` + ``popmean`` : sample and reference mean
        - ``chi_square_gof``    : ``{"observed": [...], "expected": [...]}``
        - ``chi_square_independence`` : 2D contingency table
        - ``paired``            : tuple of two paired samples (treated as groups)
        """
        if "groups" in data:
            return self.compare_groups(hypothesis, data["groups"], data.get("labels"))

        if "one_sample" in data:
            return self.compare_to_reference(hypothesis, data["one_sample"], data["popmean"])

        if "chi_square_gof" in data:
            payload = data["chi_square_gof"]
            return self.goodness_of_fit(
                hypothesis,
                payload["observed"],
                payload.get("expected"),
            )

        if "chi_square_independence" in data:
            return self.test_independence(hypothesis, data["chi_square_independence"])

        if "paired" in data:
            a, b = data["paired"]
            return self.compare_groups(hypothesis, [list(a), list(b)])

        raise ValueError(
            "Unsupported data format for evaluation. "
            "Provide one of: 'groups', 'one_sample' (with 'popmean'), "
            "'chi_square_gof', 'chi_square_independence', or 'paired'."
        )
