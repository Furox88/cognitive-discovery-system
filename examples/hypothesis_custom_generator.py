"""Minimal runnable example of a custom HypothesisGenerator implementation.

The hypothesis module exposes a HypothesisGenerator Protocol so that
researchers can supply their own generator for domain-specific logic,
data sources, or templates while reusing the rest of the structured
output model and the generate_hypotheses convenience function.

This file shows a concrete toy implementation for a narrow domain.
It is fully offline and deterministic for reproducibility in scripts
or tests.
"""

import uuid

from cds.hypothesis import (
    Domain,
    Hypothesis,
    HypothesisStatus,
    generate_hypotheses,
)


class ToyPhysicsGenerator:
    """Example custom generator satisfying the HypothesisGenerator Protocol.

    A real implementation might load templates from files, query a local
    knowledge base, apply constraints from experimental metadata, or
    delegate to an external service. The only requirement is a generate()
    method with the signature shown here.
    """

    def generate(
        self,
        research_question: str,
        domain: Domain = Domain.GENERAL_SCIENCE,
        n: int = 3,
        **kwargs,
    ) -> list[Hypothesis]:
        # Toy domain-specific ideas for illustration.
        # In practice these would come from more sophisticated sources.
        ideas = [
            "A small correction to the dispersion relation at intermediate energies accounts for the reported anomaly.",
            "An overlooked selection effect in the measurement chain produces the apparent deviation from theory.",
        ]

        hypos: list[Hypothesis] = []
        for i, idea in enumerate(ideas[:n]):
            h = Hypothesis(
                id=f"H-{uuid.uuid4().hex[:8]}",
                statement=idea,
                domain=Domain.PHYSICS,
                research_question=research_question,
                rationale=(
                    "The statement is constructed to be directly testable "
                    "against existing or planned observations."
                ),
                assumptions=[
                    "The background theoretical framework holds outside the reported regime.",
                    "Instrumental and analysis systematics have been characterized to the quoted precision.",
                ],
                predictions=[
                    "A specific functional dependence of the observable on energy (or scale) that differs from the null hypothesis by >3 sigma in a dataset of size N.",
                    "A correlation between two channels that vanishes under the standard model.",
                ],
                status=HypothesisStatus.NEW,
                confidence=0.55 + i * 0.05,
                tags=[domain.value, "custom-example"],
            )
            hypos.append(h)
        # Pad if fewer ideas than requested
        while len(hypos) < n:
            h = Hypothesis(
                id=f"H-{uuid.uuid4().hex[:8]}",
                statement=(
                    f"Variant mechanism suggested by {research_question} "
                    "yields an observable shift at accessible scales."
                ),
                domain=Domain.PHYSICS,
                research_question=research_question,
                rationale="Generic padding to satisfy the requested count.",
                assumptions=["Core model remains valid."],
                predictions=["Measurable deviation in observable O."],
                status=HypothesisStatus.NEW,
                confidence=0.4,
                tags=[domain.value, "custom-example"],
            )
            hypos.append(h)
        return hypos


if __name__ == "__main__":
    print("=== Custom HypothesisGenerator Demo ===\n")

    question = "What produces the reported discrepancy in high-energy scattering data?"

    print(f"Research question: {question}\n")

    # Default (built-in offline) generator
    print("Default offline generator:")
    default = generate_hypotheses(question, domain=Domain.PHYSICS, n=1)
    print(f"  {default[0].statement}\n")

    # Custom implementation
    print("Custom ToyPhysicsGenerator:")
    custom = ToyPhysicsGenerator()
    custom_hypos = generate_hypotheses(question, domain=Domain.PHYSICS, n=2, generator=custom)
    for h in custom_hypos:
        print(f"  ID: {h.id}")
        print(f"  Statement: {h.statement}")
        print(f"  Assumptions: {h.assumptions}")
        print(f"  Predictions (first): {h.predictions[0] if h.predictions else ''}")
        print()

    print(
        "Any object with a compatible generate() method can be passed "
        "as the generator= argument (or used directly). "
        "This keeps the rest of the hypothesis tooling unchanged."
    )
