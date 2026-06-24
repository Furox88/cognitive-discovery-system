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
from typing import TypedDict

from cds.core.models import Hypothesis, HypothesisStatus
from cds.stats.descriptive import mean as _mean
from cds.stats.descriptive import stdev as _stdev
from cds.stats.hypothesis_tests import (
    bonferroni_corrected_alpha,
    chi_square_gof,
    chi_square_independence,
    cohens_d,
    cramers_v,
    eta_squared_from_f,
    one_sample_ttest,
    one_way_anova,
    two_sample_ttest,
)


@dataclass
class EvaluationResult:
    """Detailed result of a hypothesis evaluation.

    ``effect_size`` and ``effect_size_label`` carry a standardized effect-size
    estimate (e.g. Cohen's d, eta-squared, Cramer's V) when one is defined for
    the test that was run; both are ``None`` for tests without a standard
    effect size (e.g. chi-square goodness-of-fit). Effect size is reported
    alongside the p-value but does NOT gate ``is_significant`` — a small effect
    can still be statistically significant, and that distinction is left to
    the caller to interpret.
    """

    hypothesis_id: str
    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    conclusion: str
    effect_size: float | None = None
    effect_size_label: str | None = None


class ChiSquareGofPayload(TypedDict, total=False):
    """Nested payload under the ``chi_square_gof`` dispatch key.

    ``expected`` is optional at the call site: the evaluator falls back to a
    uniform distribution over the categories when it is missing. ``total=False``
    makes both fields optional so callers can supply only ``observed``; the
    ``in``-guards in :meth:`HypothesisEvaluator.evaluate` handle presence.
    """

    observed: list[float]
    expected: list[float]


class EvaluationData(TypedDict, total=False):
    """Tagged-union payload selecting which statistical test ``evaluate`` runs.

    Exactly one of the dispatch keys below should be set; ``evaluate`` checks
    them in documented order and raises ``ValueError`` if none match. ``total=False``
    mirrors the established ``AdamState`` convention (``optimization.minimize``):
    every field is optional and presence is the dispatch signal, checked via
    ``if "<key>" in data:`` in the method body (mypy narrows those accesses).

    - ``groups``                  -> t-test (2) or ANOVA (3+); optional ``labels``
    - ``one_sample`` + ``popmean``-> one-sample t-test vs a reference mean
    - ``chi_square_gof``          -> ``{"observed": [...], "expected": [...]}``
    - ``chi_square_independence`` -> 2D contingency table
    - ``paired``                  -> tuple of two paired samples
    """

    groups: list[list[float]]
    labels: list[str]
    one_sample: list[float]
    popmean: float
    chi_square_gof: ChiSquareGofPayload
    chi_square_independence: list[list[float]]
    paired: tuple[list[float], list[float]]


class HypothesisEvaluator:
    """Autonomous evaluator that matches hypotheses with statistical tests."""

    def __init__(self, alpha: float = 0.05):
        """Store the significance threshold ``alpha`` used to flag results.

        Args:
            alpha: Two-sided significance level (default ``0.05``). A test
                with ``p < alpha`` is reported as significant.
        """
        self.alpha = alpha

    def _build_result(
        self,
        hypothesis: Hypothesis,
        test_name: str,
        statistic: float,
        p_value: float,
        effect_size: float | None = None,
        effect_size_label: str | None = None,
        alpha: float | None = None,
    ) -> EvaluationResult:
        """Format the outcome and update the hypothesis status.

        ``effect_size``/``effect_size_label`` are surfaced on the result and
        appended to the conclusion for readability but do not influence
        ``is_significant``, which remains a pure p-value-vs-alpha test.

        ``alpha`` overrides ``self.alpha`` for this result only. ``evaluate`` /
        the per-test methods leave it as ``None`` (uncorrected); only
        ``evaluate_batch`` passes the Bonferroni-corrected level so that a
        single significant result among many does not overstate evidence.
        """
        effective_alpha = self.alpha if alpha is None else alpha
        is_sig = p_value < effective_alpha
        effect_clause = ""
        if effect_size is not None and effect_size_label is not None:
            effect_clause = f" Effect size: {effect_size_label} = {effect_size:.3f}."
        if is_sig:
            conclusion = (
                f"Hypothesis supported at alpha={effective_alpha}. "
                f"Significant result found ({test_name}).{effect_clause}"
            )
            hypothesis.status = HypothesisStatus.VALIDATED
        else:
            conclusion = (
                f"Failed to support hypothesis at alpha={effective_alpha}. "
                f"No significant result ({test_name}).{effect_clause}"
            )
            hypothesis.status = HypothesisStatus.REJECTED
        return EvaluationResult(
            hypothesis_id=hypothesis.id,
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            is_significant=is_sig,
            conclusion=conclusion,
            effect_size=effect_size,
            effect_size_label=effect_size_label,
        )

    def compare_groups(
        self,
        hypothesis: Hypothesis,
        groups: list[list[float]],
        labels: list[str] | None = None,
    ) -> EvaluationResult:
        """Evaluate a hypothesis by comparing multiple numeric groups.

        Uses t-test for 2 groups, ANOVA for more. Reports Cohen's d for the
        two-group case and eta-squared for ANOVA.
        """
        if len(groups) < 2:
            raise ValueError("Evaluation requires at least 2 groups of data.")

        if len(groups) == 2:
            res = two_sample_ttest(groups[0], groups[1])
            test_name = "Two-sample t-test"
            # Cohen's d standardizes the mean difference by the pooled SD.
            try:
                eff = cohens_d(groups[0], groups[1])
                eff_label = "Cohen's d"
            except ValueError:  # pragma: no cover - unreachable: two_sample_ttest
                # above raises on the same zero-variance / short-sample inputs
                # that cohens_d would, so control never reaches this handler.
                eff, eff_label = None, None
        else:
            res = one_way_anova(*groups)
            test_name = "One-way ANOVA"
            # eta^2 = (F * df1) / (F * df1 + df2); df1 = k-1, df2 = N-k.
            df1 = len(groups) - 1
            n_total = sum(len(g) for g in groups)
            df2 = n_total - len(groups)
            eff = eta_squared_from_f(res.statistic, df1, df2)
            eff_label = "eta-squared"

        return self._build_result(hypothesis, test_name, res.statistic, res.p_value, eff, eff_label)

    def compare_to_reference(
        self,
        hypothesis: Hypothesis,
        sample: list[float],
        popmean: float,
    ) -> EvaluationResult:
        """One-sample t-test: does the sample differ from a reference mean?

        Reports a one-sample Cohen's d = (mean(sample) - popmean) / stdev(sample),
        standardizing the shift by the sample's own standard deviation.
        """
        if len(sample) < 2:
            raise ValueError("One-sample evaluation requires at least 2 observations.")
        res = one_sample_ttest(sample, popmean)
        try:
            s = _stdev(sample, ddof=1)
            eff = (_mean(sample) - popmean) / s if s > 0 else None
            eff_label = "Cohen's d" if eff is not None else None
        except ValueError:  # pragma: no cover - unreachable: len(sample)>=2 guard
            # above guarantees _stdev has enough points, so it never raises here.
            eff, eff_label = None, None
        return self._build_result(
            hypothesis, "One-sample t-test", res.statistic, res.p_value, eff, eff_label
        )

    def goodness_of_fit(
        self,
        hypothesis: Hypothesis,
        observed: list[float],
        expected: list[float] | None = None,
    ) -> EvaluationResult:
        """Chi-square goodness-of-fit: observed vs expected category counts.

        If ``expected`` is omitted, a uniform distribution over the categories
        is assumed (all categories equally likely). No standardized effect
        size is reported — chi-square GOF has no single accepted one
        (standardized residuals would be needed), so the fields stay ``None``.
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
        """Chi-square test of independence on a contingency table.

        Reports Cramer's V, the chi-square statistic normalized to [0, 1] by
        the sample size and the smaller table dimension.
        """
        if len(table) < 2 or any(len(row) < 2 for row in table):
            raise ValueError("Independence test requires a 2x2 or larger contingency table.")
        res = chi_square_independence(table)
        try:
            eff = cramers_v(table)
            eff_label = "Cramer's V"
        except ValueError:  # pragma: no cover - unreachable: the 2x2/rectangular
            # guard above and chi_square_independence's grand-total check cover
            # every input cramers_v would reject, so this handler never runs.
            eff, eff_label = None, None
        return self._build_result(
            hypothesis, "Chi-square independence", res.statistic, res.p_value, eff, eff_label
        )

    def evaluate(self, hypothesis: Hypothesis, data: EvaluationData) -> EvaluationResult:
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

    def evaluate_batch(
        self,
        hypotheses: list[Hypothesis],
        data_items: list[EvaluationData],
    ) -> list[EvaluationResult]:
        """Evaluate a family of hypotheses with a multiple-comparison correction.

        Runs each (hypothesis, data) pair through :meth:`evaluate` and then
        re-judges significance against a Bonferroni-corrected threshold
        (``self.alpha / k``) so that a single spurious p-value among many does
        not overstate the family's evidence. ``k`` is the number of
        comparisons (``len(hypotheses)``).

        The first pass evaluates each hypothesis with the uncorrected
        ``self.alpha`` (matching :meth:`evaluate`). The second pass rebuilds
        the results against the corrected alpha, so a hypothesis that was
        marginally significant alone (e.g. p=0.04 < 0.05) may be REJECTED
        once the family is considered (0.04 > 0.05/3 ~= 0.0167).

        Args:
            hypotheses: k hypotheses to evaluate, in order.
            data_items: k matching data payloads (same length, same order).

        Returns:
            One :class:`EvaluationResult` per hypothesis, judged at the
            Bonferroni-corrected alpha. Each carries the test statistic,
            p-value, and effect size as usual; only ``is_significant`` and the
            conclusion reflect the corrected threshold.

        Raises:
            ValueError: if ``hypotheses`` and ``data_items`` differ in length,
                or the family is empty.
        """
        if len(hypotheses) != len(data_items):
            raise ValueError("hypotheses and data_items must have the same length")
        if not hypotheses:
            raise ValueError("evaluate_batch requires at least one hypothesis")

        # First pass: per-hypothesis results at the uncorrected alpha. We need
        # the raw (statistic, p_value, effect size) so we can re-judge without
        # re-running the tests.
        raw_results = [self.evaluate(h, d) for h, d in zip(hypotheses, data_items)]

        # Bonferroni-corrected per-test alpha for this family of k tests.
        corrected_alpha = bonferroni_corrected_alpha(self.alpha, len(hypotheses))

        # Second pass: rebuild each result against the corrected alpha. Reuse
        # _build_result so the conclusion text and status stay consistent with
        # the single-hypothesis path; the hypothesis status set in the first
        # pass is overwritten here.
        corrected: list[EvaluationResult] = []
        for h, raw in zip(hypotheses, raw_results):
            rebuilt = self._build_result(
                h,
                raw.test_name,
                raw.statistic,
                raw.p_value,
                raw.effect_size,
                raw.effect_size_label,
                alpha=corrected_alpha,
            )
            corrected.append(rebuilt)
        return corrected
