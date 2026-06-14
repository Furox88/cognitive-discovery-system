"""
Evaluator for scientific hypotheses using empirical data.

This module provides the connection between generated hypotheses and
statistical verification, enabling an autonomous discovery loop.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cds.core.models import Hypothesis, HypothesisStatus
from cds.stats.hypothesis_tests import (
    one_way_anova,
    two_sample_ttest,
    TestResult,
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

    def compare_groups(
        self, 
        hypothesis: Hypothesis, 
        groups: list[list[float]], 
        labels: list[str] | None = None
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

        is_sig = res.p_value < self.alpha
        
        if is_sig:
            conclusion = f"Hypothesis supported at alpha={self.alpha}. Significant difference found."
            hypothesis.status = HypothesisStatus.VALIDATED
        else:
            conclusion = f"Failed to support hypothesis at alpha={self.alpha}. No significant difference."
            hypothesis.status = HypothesisStatus.REJECTED

        return EvaluationResult(
            hypothesis_id=hypothesis.id,
            test_name=test_name,
            statistic=res.statistic,
            p_value=res.p_value,
            is_significant=is_sig,
            conclusion=conclusion
        )

    def evaluate(self, hypothesis: Hypothesis, data: dict[str, Any]) -> EvaluationResult:
        """General evaluation entry point (extensible)."""
        # In a real AI-driven system, this would parse the hypothesis statement
        # to determine the correct test. Here we provide a structured interface.
        if "groups" in data:
            return self.compare_groups(hypothesis, data["groups"], data.get("labels"))
        
        raise NotImplementedError("Evaluation logic for this data format is not yet implemented.")
